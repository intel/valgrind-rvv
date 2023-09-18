
/*--------------------------------------------------------------------*/
/*--- begin                                   guest_riscv64_toIR.c ---*/
/*--------------------------------------------------------------------*/

/*
   This file is part of Valgrind, a dynamic binary instrumentation
   framework.

   Copyright (C) 2020-2023 Petr Pavlu
      petr.pavlu@dagobah.cz

   This program is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see <http://www.gnu.org/licenses/>.

   The GNU General Public License is contained in the file COPYING.
*/

/* Translates riscv64 code to IR. */

/* "Special" instructions.

   This instruction decoder can decode four special instructions which mean
   nothing natively (are no-ops as far as regs/mem are concerned) but have
   meaning for supporting Valgrind. A special instruction is flagged by
   a 16-byte preamble:

      00305013 00d05013 03305013 03d05013
      (srli zero, zero, 3;   srli zero, zero, 13
       srli zero, zero, 51;  srli zero, zero, 61)

   Following that, one of the following 4 are allowed (standard interpretation
   in parentheses):

      00a56533 (or a0, a0, a0)   a3 = client_request ( a4 )
      00b5e5b3 (or a1, a1, a1)   a3 = guest_NRADDR
      00c66633 (or a2, a2, a2)   branch-and-link-to-noredir t0
      00d6e6b3 (or a3, a3, a3)   IR injection

   Any other bytes following the 16-byte preamble are illegal and constitute
   a failure in instruction decoding. This all assumes that the preamble will
   never occur except in specific code fragments designed for Valgrind to catch.
*/

#include "libvex_guest_riscv64.h"

#include "guest_riscv64_defs.h"
#include "main_globals.h"
#include "main_util.h"

#include "coregrind/pub_core_transtab.h"

/*------------------------------------------------------------*/
/*--- Debugging output                                     ---*/
/*------------------------------------------------------------*/

#define DIP(format, args...)                                                   \
   do {                                                                        \
      if (vex_traceflags & VEX_TRACE_FE)                                       \
         vex_printf(format, ##args);                                           \
   } while (0)

#define DIS(buf, format, args...)                                              \
   do {                                                                        \
      if (vex_traceflags & VEX_TRACE_FE)                                       \
         vex_sprintf(buf, format, ##args);                                     \
   } while (0)

/*------------------------------------------------------------*/
/*--- Helper bits and pieces for deconstructing the        ---*/
/*--- riscv64 insn stream.                                 ---*/
/*------------------------------------------------------------*/

/* Do a little-endian load of a 32-bit word, regardless of the endianness of the
   underlying host. */
static inline UInt getUIntLittleEndianly(const UChar* p)
{
   UInt w = 0;
   w      = (w << 8) | p[3];
   w      = (w << 8) | p[2];
   w      = (w << 8) | p[1];
   w      = (w << 8) | p[0];
   return w;
}

/* Do read of an instruction, which can be 16-bit (compressed) or 32-bit in
   size. */
static inline UInt getInsn(const UChar* p)
{
   Bool is_compressed = (p[0] & 0x3) != 0x3;
   UInt w             = 0;
   if (!is_compressed) {
      w = (w << 8) | p[3];
      w = (w << 8) | p[2];
   }
   w = (w << 8) | p[1];
   w = (w << 8) | p[0];
   return w;
}

/* Produce _uint[_bMax:_bMin]. */
#define SLICE_UInt(_uint, _bMax, _bMin)                                        \
   ((((UInt)(_uint)) >> (_bMin)) &                                             \
    (UInt)((1ULL << ((_bMax) - (_bMin) + 1)) - 1ULL))

/*------------------------------------------------------------*/
/*--- Helpers for constructing IR.                         ---*/
/*------------------------------------------------------------*/

/* Create an expression to produce a 64-bit constant. */
static IRExpr* mkU64(ULong i) { return IRExpr_Const(IRConst_U64(i)); }

/* Create an expression to produce a 32-bit constant. */
static IRExpr* mkU32(UInt i) { return IRExpr_Const(IRConst_U32(i)); }

static IRExpr* mkU16(UInt i) { return IRExpr_Const(IRConst_U16((UShort)i)); }

/* Create an expression to produce an 8-bit constant. */
static IRExpr* mkU8(UInt i)
{
   vassert(i < 256);
   return IRExpr_Const(IRConst_U8((UChar)i));
}

static IRExpr* mkU(IRType ty, ULong i)
{
   switch (ty) {
   case Ity_I8:   return mkU8((UChar)i);
   case Ity_I16:  return mkU16((UShort)i);
   case Ity_I32:  return mkU32((UInt)i);
   case Ity_I64:  return mkU64(i);
   default:       vassert(0);
   }
}

/* Create an expression to read a temporary. */
static IRExpr* mkexpr(IRTemp tmp) { return IRExpr_RdTmp(tmp); }

/* Create an unary-operation expression. */
static IRExpr* unop(IROp op, IRExpr* a) { return IRExpr_Unop(op, a); }

/* Create a binary-operation expression. */
static IRExpr* binop(IROp op, IRExpr* a1, IRExpr* a2)
{
   return IRExpr_Binop(op, a1, a2);
}

/* Create a ternary-operation expression. */
static IRExpr* triop(IROp op, IRExpr* a1, IRExpr* a2, IRExpr* a3)
{
   return IRExpr_Triop(op, a1, a2, a3);
}

/* Create a quaternary-operation expression. */
static IRExpr* qop(IROp op, IRExpr* a1, IRExpr* a2, IRExpr* a3, IRExpr* a4)
{
   return IRExpr_Qop(op, a1, a2, a3, a4);
}

/* Create an expression to load a value from memory (in the little-endian
   order). */
static IRExpr* loadLE(IRType ty, IRExpr* addr)
{
   return IRExpr_Load(Iend_LE, ty, addr);
}

/* Add a statement to the list held by irsb. */
static void stmt(/*MOD*/ IRSB* irsb, IRStmt* st) { addStmtToIRSB(irsb, st); }

/* Add a statement to assign a value to a temporary. */
static void assign(/*MOD*/ IRSB* irsb, IRTemp dst, IRExpr* e)
{
   stmt(irsb, IRStmt_WrTmp(dst, e));
}

/* Generate a statement to store a value in memory (in the little-endian
   order). */
static void storeLE(/*MOD*/ IRSB* irsb, IRExpr* addr, IRExpr* data)
{
   stmt(irsb, IRStmt_Store(Iend_LE, addr, data));
}

/* Generate a new temporary of the given type. */
static IRTemp newTemp(/*MOD*/ IRSB* irsb, IRType ty)
{
   vassert(isPlausibleIRType(ty));
   return newIRTemp(irsb->tyenv, ty);
}

/* Sign-extend a 32/64-bit integer expression to 64 bits. */
static IRExpr* widenSto64(IRType srcTy, IRExpr* e)
{
   switch (srcTy) {
   case Ity_I64:
      return e;
   case Ity_I32:
      return unop(Iop_32Sto64, e);
   case Ity_I16:
      return unop(Iop_16Sto64, e);
   case Ity_I8:
      return unop(Iop_8Sto64, e);
   default:
      vpanic("widenSto64(riscv64)");
   }
}

/* Narrow a 64-bit integer expression to 32/64 bits. */
static IRExpr* narrowFrom64(IRType dstTy, IRExpr* e)
{
   switch (dstTy) {
   case Ity_I64:
      return e;
   case Ity_I32:
      return unop(Iop_64to32, e);
   case Ity_I16:
      return unop(Iop_64to16, e);
   case Ity_I8:
      return unop(Iop_64to8, e);
   default:
      vpanic("narrowFrom64(riscv64)");
   }
}

/*------------------------------------------------------------*/
/*--- Offsets of various parts of the riscv64 guest state  ---*/
/*------------------------------------------------------------*/

#define OFFB_X0  offsetof(VexGuestRISCV64State, guest_x0)
#define OFFB_X1  offsetof(VexGuestRISCV64State, guest_x1)
#define OFFB_X2  offsetof(VexGuestRISCV64State, guest_x2)
#define OFFB_X3  offsetof(VexGuestRISCV64State, guest_x3)
#define OFFB_X4  offsetof(VexGuestRISCV64State, guest_x4)
#define OFFB_X5  offsetof(VexGuestRISCV64State, guest_x5)
#define OFFB_X6  offsetof(VexGuestRISCV64State, guest_x6)
#define OFFB_X7  offsetof(VexGuestRISCV64State, guest_x7)
#define OFFB_X8  offsetof(VexGuestRISCV64State, guest_x8)
#define OFFB_X9  offsetof(VexGuestRISCV64State, guest_x9)
#define OFFB_X10 offsetof(VexGuestRISCV64State, guest_x10)
#define OFFB_X11 offsetof(VexGuestRISCV64State, guest_x11)
#define OFFB_X12 offsetof(VexGuestRISCV64State, guest_x12)
#define OFFB_X13 offsetof(VexGuestRISCV64State, guest_x13)
#define OFFB_X14 offsetof(VexGuestRISCV64State, guest_x14)
#define OFFB_X15 offsetof(VexGuestRISCV64State, guest_x15)
#define OFFB_X16 offsetof(VexGuestRISCV64State, guest_x16)
#define OFFB_X17 offsetof(VexGuestRISCV64State, guest_x17)
#define OFFB_X18 offsetof(VexGuestRISCV64State, guest_x18)
#define OFFB_X19 offsetof(VexGuestRISCV64State, guest_x19)
#define OFFB_X20 offsetof(VexGuestRISCV64State, guest_x20)
#define OFFB_X21 offsetof(VexGuestRISCV64State, guest_x21)
#define OFFB_X22 offsetof(VexGuestRISCV64State, guest_x22)
#define OFFB_X23 offsetof(VexGuestRISCV64State, guest_x23)
#define OFFB_X24 offsetof(VexGuestRISCV64State, guest_x24)
#define OFFB_X25 offsetof(VexGuestRISCV64State, guest_x25)
#define OFFB_X26 offsetof(VexGuestRISCV64State, guest_x26)
#define OFFB_X27 offsetof(VexGuestRISCV64State, guest_x27)
#define OFFB_X28 offsetof(VexGuestRISCV64State, guest_x28)
#define OFFB_X29 offsetof(VexGuestRISCV64State, guest_x29)
#define OFFB_X30 offsetof(VexGuestRISCV64State, guest_x30)
#define OFFB_X31 offsetof(VexGuestRISCV64State, guest_x31)
#define OFFB_PC  offsetof(VexGuestRISCV64State, guest_pc)

#define OFFB_F0   offsetof(VexGuestRISCV64State, guest_f0)
#define OFFB_F1   offsetof(VexGuestRISCV64State, guest_f1)
#define OFFB_F2   offsetof(VexGuestRISCV64State, guest_f2)
#define OFFB_F3   offsetof(VexGuestRISCV64State, guest_f3)
#define OFFB_F4   offsetof(VexGuestRISCV64State, guest_f4)
#define OFFB_F5   offsetof(VexGuestRISCV64State, guest_f5)
#define OFFB_F6   offsetof(VexGuestRISCV64State, guest_f6)
#define OFFB_F7   offsetof(VexGuestRISCV64State, guest_f7)
#define OFFB_F8   offsetof(VexGuestRISCV64State, guest_f8)
#define OFFB_F9   offsetof(VexGuestRISCV64State, guest_f9)
#define OFFB_F10  offsetof(VexGuestRISCV64State, guest_f10)
#define OFFB_F11  offsetof(VexGuestRISCV64State, guest_f11)
#define OFFB_F12  offsetof(VexGuestRISCV64State, guest_f12)
#define OFFB_F13  offsetof(VexGuestRISCV64State, guest_f13)
#define OFFB_F14  offsetof(VexGuestRISCV64State, guest_f14)
#define OFFB_F15  offsetof(VexGuestRISCV64State, guest_f15)
#define OFFB_F16  offsetof(VexGuestRISCV64State, guest_f16)
#define OFFB_F17  offsetof(VexGuestRISCV64State, guest_f17)
#define OFFB_F18  offsetof(VexGuestRISCV64State, guest_f18)
#define OFFB_F19  offsetof(VexGuestRISCV64State, guest_f19)
#define OFFB_F20  offsetof(VexGuestRISCV64State, guest_f20)
#define OFFB_F21  offsetof(VexGuestRISCV64State, guest_f21)
#define OFFB_F22  offsetof(VexGuestRISCV64State, guest_f22)
#define OFFB_F23  offsetof(VexGuestRISCV64State, guest_f23)
#define OFFB_F24  offsetof(VexGuestRISCV64State, guest_f24)
#define OFFB_F25  offsetof(VexGuestRISCV64State, guest_f25)
#define OFFB_F26  offsetof(VexGuestRISCV64State, guest_f26)
#define OFFB_F27  offsetof(VexGuestRISCV64State, guest_f27)
#define OFFB_F28  offsetof(VexGuestRISCV64State, guest_f28)
#define OFFB_F29  offsetof(VexGuestRISCV64State, guest_f29)
#define OFFB_F30  offsetof(VexGuestRISCV64State, guest_f30)
#define OFFB_F31  offsetof(VexGuestRISCV64State, guest_f31)
#define OFFB_FCSR offsetof(VexGuestRISCV64State, guest_fcsr)

#define OFFB_EMNOTE  offsetof(VexGuestRISCV64State, guest_EMNOTE)
#define OFFB_CMSTART offsetof(VexGuestRISCV64State, guest_CMSTART)
#define OFFB_CMLEN   offsetof(VexGuestRISCV64State, guest_CMLEN)
#define OFFB_NRADDR  offsetof(VexGuestRISCV64State, guest_NRADDR)

#define OFFB_LLSC_SIZE offsetof(VexGuestRISCV64State, guest_LLSC_SIZE)
#define OFFB_LLSC_ADDR offsetof(VexGuestRISCV64State, guest_LLSC_ADDR)
#define OFFB_LLSC_DATA offsetof(VexGuestRISCV64State, guest_LLSC_DATA)

#define OFFB_V0   offsetof(VexGuestRISCV64State, guest_v0)
#define OFFB_V1   offsetof(VexGuestRISCV64State, guest_v1)
#define OFFB_V2   offsetof(VexGuestRISCV64State, guest_v2)
#define OFFB_V3   offsetof(VexGuestRISCV64State, guest_v3)
#define OFFB_V4   offsetof(VexGuestRISCV64State, guest_v4)
#define OFFB_V5   offsetof(VexGuestRISCV64State, guest_v5)
#define OFFB_V6   offsetof(VexGuestRISCV64State, guest_v6)
#define OFFB_V7   offsetof(VexGuestRISCV64State, guest_v7)
#define OFFB_V8   offsetof(VexGuestRISCV64State, guest_v8)
#define OFFB_V9   offsetof(VexGuestRISCV64State, guest_v9)
#define OFFB_V10  offsetof(VexGuestRISCV64State, guest_v10)
#define OFFB_V11  offsetof(VexGuestRISCV64State, guest_v11)
#define OFFB_V12  offsetof(VexGuestRISCV64State, guest_v12)
#define OFFB_V13  offsetof(VexGuestRISCV64State, guest_v13)
#define OFFB_V14  offsetof(VexGuestRISCV64State, guest_v14)
#define OFFB_V15  offsetof(VexGuestRISCV64State, guest_v15)
#define OFFB_V16  offsetof(VexGuestRISCV64State, guest_v16)
#define OFFB_V17  offsetof(VexGuestRISCV64State, guest_v17)
#define OFFB_V18  offsetof(VexGuestRISCV64State, guest_v18)
#define OFFB_V19  offsetof(VexGuestRISCV64State, guest_v19)
#define OFFB_V20  offsetof(VexGuestRISCV64State, guest_v20)
#define OFFB_V21  offsetof(VexGuestRISCV64State, guest_v21)
#define OFFB_V22  offsetof(VexGuestRISCV64State, guest_v22)
#define OFFB_V23  offsetof(VexGuestRISCV64State, guest_v23)
#define OFFB_V24  offsetof(VexGuestRISCV64State, guest_v24)
#define OFFB_V25  offsetof(VexGuestRISCV64State, guest_v25)
#define OFFB_V26  offsetof(VexGuestRISCV64State, guest_v26)
#define OFFB_V27  offsetof(VexGuestRISCV64State, guest_v27)
#define OFFB_V28  offsetof(VexGuestRISCV64State, guest_v28)
#define OFFB_V29  offsetof(VexGuestRISCV64State, guest_v29)
#define OFFB_V30  offsetof(VexGuestRISCV64State, guest_v30)
#define OFFB_V31  offsetof(VexGuestRISCV64State, guest_v31)

#define OFFB_VL    offsetof(VexGuestRISCV64State, guest_vl)
#define OFFB_VTYPE offsetof(VexGuestRISCV64State, guest_vtype)

/*------------------------------------------------------------*/
/*--- Integer registers                                    ---*/
/*------------------------------------------------------------*/

static Int offsetIReg64(UInt iregNo)
{
   switch (iregNo) {
   case 0:
      return OFFB_X0;
   case 1:
      return OFFB_X1;
   case 2:
      return OFFB_X2;
   case 3:
      return OFFB_X3;
   case 4:
      return OFFB_X4;
   case 5:
      return OFFB_X5;
   case 6:
      return OFFB_X6;
   case 7:
      return OFFB_X7;
   case 8:
      return OFFB_X8;
   case 9:
      return OFFB_X9;
   case 10:
      return OFFB_X10;
   case 11:
      return OFFB_X11;
   case 12:
      return OFFB_X12;
   case 13:
      return OFFB_X13;
   case 14:
      return OFFB_X14;
   case 15:
      return OFFB_X15;
   case 16:
      return OFFB_X16;
   case 17:
      return OFFB_X17;
   case 18:
      return OFFB_X18;
   case 19:
      return OFFB_X19;
   case 20:
      return OFFB_X20;
   case 21:
      return OFFB_X21;
   case 22:
      return OFFB_X22;
   case 23:
      return OFFB_X23;
   case 24:
      return OFFB_X24;
   case 25:
      return OFFB_X25;
   case 26:
      return OFFB_X26;
   case 27:
      return OFFB_X27;
   case 28:
      return OFFB_X28;
   case 29:
      return OFFB_X29;
   case 30:
      return OFFB_X30;
   case 31:
      return OFFB_X31;
   default:
      vassert(0);
   }
}

/* Obtain ABI name of a register. */
static const HChar* nameIReg(UInt iregNo)
{
   vassert(iregNo < 32);
   static const HChar* names[32] = {
      "zero", "ra", "sp", "gp", "tp",  "t0",  "t1", "t2", "s0", "s1", "a0",
      "a1",   "a2", "a3", "a4", "a5",  "a6",  "a7", "s2", "s3", "s4", "s5",
      "s6",   "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"};
   return names[iregNo];
}

/* Read a 64-bit value from a guest integer register. */
static IRExpr* getIReg64(UInt iregNo)
{
   vassert(iregNo < 32);
   return IRExpr_Get(offsetIReg64(iregNo), Ity_I64);
}

/* Write a 64-bit value into a guest integer register. */
static void putIReg64(/*OUT*/ IRSB* irsb, UInt iregNo, /*IN*/ IRExpr* e)
{
   vassert(iregNo > 0 && iregNo < 32);
   vassert(typeOfIRExpr(irsb->tyenv, e) == Ity_I64);
   stmt(irsb, IRStmt_Put(offsetIReg64(iregNo), e));
}

/* Read a 32-bit value from a guest integer register. */
static IRExpr* getIReg32(UInt iregNo)
{
   vassert(iregNo < 32);
   return unop(Iop_64to32, IRExpr_Get(offsetIReg64(iregNo), Ity_I64));
}

/* Write a 32-bit value into a guest integer register. */
static void putIReg32(/*OUT*/ IRSB* irsb, UInt iregNo, /*IN*/ IRExpr* e)
{
   vassert(iregNo > 0 && iregNo < 32);
   vassert(typeOfIRExpr(irsb->tyenv, e) == Ity_I32);
   stmt(irsb, IRStmt_Put(offsetIReg64(iregNo), unop(Iop_32Sto64, e)));
}

/* Write an address into the guest pc. */
static void putPC(/*OUT*/ IRSB* irsb, /*IN*/ IRExpr* e)
{
   vassert(typeOfIRExpr(irsb->tyenv, e) == Ity_I64);
   stmt(irsb, IRStmt_Put(OFFB_PC, e));
}

/*------------------------------------------------------------*/
/*--- Vector registers                                     ---*/
/*------------------------------------------------------------*/
static Int offsetVReg(UInt vregNo)
{
   switch (vregNo) {
   case 0:
      return OFFB_V0;
   case 1:
      return OFFB_V1;
   case 2:
      return OFFB_V2;
   case 3:
      return OFFB_V3;
   case 4:
      return OFFB_V4;
   case 5:
      return OFFB_V5;
   case 6:
      return OFFB_V6;
   case 7:
      return OFFB_V7;
   case 8:
      return OFFB_V8;
   case 9:
      return OFFB_V9;
   case 10:
      return OFFB_V10;
   case 11:
      return OFFB_V11;
   case 12:
      return OFFB_V12;
   case 13:
      return OFFB_V13;
   case 14:
      return OFFB_V14;
   case 15:
      return OFFB_V15;
   case 16:
      return OFFB_V16;
   case 17:
      return OFFB_V17;
   case 18:
      return OFFB_V18;
   case 19:
      return OFFB_V19;
   case 20:
      return OFFB_V20;
   case 21:
      return OFFB_V21;
   case 22:
      return OFFB_V22;
   case 23:
      return OFFB_V23;
   case 24:
      return OFFB_V24;
   case 25:
      return OFFB_V25;
   case 26:
      return OFFB_V26;
   case 27:
      return OFFB_V27;
   case 28:
      return OFFB_V28;
   case 29:
      return OFFB_V29;
   case 30:
      return OFFB_V30;
   case 31:
      return OFFB_V31;
   default:
      vassert(0);
   }
}

static const HChar* nameVReg(UInt iregNo)
{
   vassert(iregNo < 32);
   static const HChar* names[32] = {
       "v0",  "v1",  "v2",  "v3",  "v4",  "v5",  "v6",  "v7",
       "v8",  "v9", "v10", "v11", "v12", "v13", "v14", "v15",
      "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23",
      "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31"};
   return names[iregNo];
}

static IRExpr* getVReg(UInt vregNo, UInt offset, IRType ty)
{
   vassert(vregNo < 32);
   return IRExpr_Get(offsetVReg(vregNo) + offset, ty);
}

static void putVReg(/*OUT*/ IRSB* irsb, UInt vregNo, UInt offset, /*IN*/ IRExpr* e)
{
   vassert(vregNo < 32);
   stmt(irsb, IRStmt_Put(offsetVReg(vregNo) + offset, e));
}


/*------------------------------------------------------------*/
/*--- Floating-point registers                             ---*/
/*------------------------------------------------------------*/

static Int offsetFReg(UInt fregNo)
{
   switch (fregNo) {
   case 0:
      return OFFB_F0;
   case 1:
      return OFFB_F1;
   case 2:
      return OFFB_F2;
   case 3:
      return OFFB_F3;
   case 4:
      return OFFB_F4;
   case 5:
      return OFFB_F5;
   case 6:
      return OFFB_F6;
   case 7:
      return OFFB_F7;
   case 8:
      return OFFB_F8;
   case 9:
      return OFFB_F9;
   case 10:
      return OFFB_F10;
   case 11:
      return OFFB_F11;
   case 12:
      return OFFB_F12;
   case 13:
      return OFFB_F13;
   case 14:
      return OFFB_F14;
   case 15:
      return OFFB_F15;
   case 16:
      return OFFB_F16;
   case 17:
      return OFFB_F17;
   case 18:
      return OFFB_F18;
   case 19:
      return OFFB_F19;
   case 20:
      return OFFB_F20;
   case 21:
      return OFFB_F21;
   case 22:
      return OFFB_F22;
   case 23:
      return OFFB_F23;
   case 24:
      return OFFB_F24;
   case 25:
      return OFFB_F25;
   case 26:
      return OFFB_F26;
   case 27:
      return OFFB_F27;
   case 28:
      return OFFB_F28;
   case 29:
      return OFFB_F29;
   case 30:
      return OFFB_F30;
   case 31:
      return OFFB_F31;
   default:
      vassert(0);
   }
}

/* Obtain ABI name of a register. */
static const HChar* nameFReg(UInt fregNo)
{
   vassert(fregNo < 32);
   static const HChar* names[32] = {
      "ft0", "ft1", "ft2",  "ft3",  "ft4", "ft5", "ft6",  "ft7",
      "fs0", "fs1", "fa0",  "fa1",  "fa2", "fa3", "fa4",  "fa5",
      "fa6", "fa7", "fs2",  "fs3",  "fs4", "fs5", "fs6",  "fs7",
      "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11"};
   return names[fregNo];
}

/* Read a 64-bit value from a guest floating-point register. */
static IRExpr* getFReg64(UInt fregNo)
{
   vassert(fregNo < 32);
   return IRExpr_Get(offsetFReg(fregNo), Ity_F64);
}

/* Write a 64-bit value into a guest floating-point register. */
static void putFReg64(/*OUT*/ IRSB* irsb, UInt fregNo, /*IN*/ IRExpr* e)
{
   vassert(fregNo < 32);
   vassert(typeOfIRExpr(irsb->tyenv, e) == Ity_F64);
   stmt(irsb, IRStmt_Put(offsetFReg(fregNo), e));
}

/* Read a 32-bit value from a guest floating-point register. */
static IRExpr* getFReg32(UInt fregNo)
{
   vassert(fregNo < 32);
   /* Note that the following access depends on the host being little-endian
      which is checked in disInstr_RISCV64(). */
   /* TODO Check that the value is correctly NaN-boxed. If not then return
      the 32-bit canonical qNaN, as mandated by the RISC-V ISA. */
   return IRExpr_Get(offsetFReg(fregNo), Ity_F32);
}

/* Write a 32-bit value into a guest floating-point register. */
static void putFReg32(/*OUT*/ IRSB* irsb, UInt fregNo, /*IN*/ IRExpr* e)
{
   vassert(fregNo < 32);
   vassert(typeOfIRExpr(irsb->tyenv, e) == Ity_F32);
   /* Note that the following access depends on the host being little-endian
      which is checked in disInstr_RISCV64(). */
   Int offset = offsetFReg(fregNo);
   stmt(irsb, IRStmt_Put(offset, e));
   /* Write 1's in the upper bits of the target 64-bit register to create
      a NaN-boxed value, as mandated by the RISC-V ISA. */
   stmt(irsb, IRStmt_Put(offset + 4, mkU32(0xffffffff)));
   /* TODO Check that this works with Memcheck. */
}

/* Read a 32-bit value from the fcsr. */
static IRExpr* getFCSR(void) { return IRExpr_Get(OFFB_FCSR, Ity_I32); }

/* Write a 32-bit value into the fcsr. */
static void putFCSR(/*OUT*/ IRSB* irsb, /*IN*/ IRExpr* e)
{
   vassert(typeOfIRExpr(irsb->tyenv, e) == Ity_I32);
   stmt(irsb, IRStmt_Put(OFFB_FCSR, e));
}

/* Accumulate exception flags in fcsr. */
static void accumulateFFLAGS(/*OUT*/ IRSB* irsb, /*IN*/ IRExpr* e)
{
   vassert(typeOfIRExpr(irsb->tyenv, e) == Ity_I32);
   putFCSR(irsb, binop(Iop_Or32, getFCSR(), binop(Iop_And32, e, mkU32(0x1f))));
}

/* Generate IR to get hold of the rounding mode in both RISC-V and IR
   formats. A floating-point operation can use either a static rounding mode
   encoded in the instruction, or a dynamic rounding mode held in fcsr. Bind the
   final result to the passed temporaries (which are allocated by the function).
 */
static void mk_get_rounding_mode(/*MOD*/ IRSB*   irsb,
                                 /*OUT*/ IRTemp* rm_RISCV,
                                 /*OUT*/ IRTemp* rm_IR,
                                 UInt            inst_rm_RISCV)
{
   /*
      rounding mode                | RISC-V |  IR
      --------------------------------------------
      to nearest, ties to even     |   000  | 0000
      to zero                      |   001  | 0011
      to +infinity                 |   010  | 0010
      to -infinity                 |   011  | 0001
      to nearest, ties away from 0 |   100  | 0100
      invalid                      |   101  | 1000
      invalid                      |   110  | 1000
      dynamic                      |   111  | 1000

      The 'dynamic' value selects the mode from fcsr. Its value is valid when
      encoded in the instruction but naturally invalid when found in fcsr.

      Static mode is known at the decode time and can be directly expressed by
      a respective rounding mode IR constant.

      Dynamic mode requires a runtime mapping from the RISC-V to the IR mode.
      It can be implemented using the following transformation:
         t0 = fcsr_rm_RISCV - 20
         t1 = t0 >> 2
         t2 = fcsr_rm_RISCV + 3
         t3 = t2 ^ 3
         rm_IR = t1 & t3
   */
   *rm_RISCV = newTemp(irsb, Ity_I32);
   *rm_IR    = newTemp(irsb, Ity_I32);
   switch (inst_rm_RISCV) {
   case 0b000:
      assign(irsb, *rm_RISCV, mkU32(0));
      assign(irsb, *rm_IR, mkU32(Irrm_NEAREST));
      break;
   case 0b001:
      assign(irsb, *rm_RISCV, mkU32(1));
      assign(irsb, *rm_IR, mkU32(Irrm_ZERO));
      break;
   case 0b010:
      assign(irsb, *rm_RISCV, mkU32(2));
      assign(irsb, *rm_IR, mkU32(Irrm_PosINF));
      break;
   case 0b011:
      assign(irsb, *rm_RISCV, mkU32(3));
      assign(irsb, *rm_IR, mkU32(Irrm_NegINF));
      break;
   case 0b100:
      assign(irsb, *rm_RISCV, mkU32(4));
      assign(irsb, *rm_IR, mkU32(Irrm_NEAREST_TIE_AWAY_0));
      break;
   case 0b101:
      assign(irsb, *rm_RISCV, mkU32(5));
      assign(irsb, *rm_IR, mkU32(Irrm_INVALID));
      break;
   case 0b110:
      assign(irsb, *rm_RISCV, mkU32(6));
      assign(irsb, *rm_IR, mkU32(Irrm_INVALID));
      break;
   case 0b111: {
      assign(irsb, *rm_RISCV,
             binop(Iop_And32, binop(Iop_Shr32, getFCSR(), mkU8(5)), mkU32(7)));
      IRTemp t0 = newTemp(irsb, Ity_I32);
      assign(irsb, t0, binop(Iop_Sub32, mkexpr(*rm_RISCV), mkU32(20)));
      IRTemp t1 = newTemp(irsb, Ity_I32);
      assign(irsb, t1, binop(Iop_Shr32, mkexpr(t0), mkU8(2)));
      IRTemp t2 = newTemp(irsb, Ity_I32);
      assign(irsb, t2, binop(Iop_Add32, mkexpr(*rm_RISCV), mkU32(3)));
      IRTemp t3 = newTemp(irsb, Ity_I32);
      assign(irsb, t3, binop(Iop_Xor32, mkexpr(t2), mkU32(3)));
      assign(irsb, *rm_IR, binop(Iop_And32, mkexpr(t1), mkexpr(t3)));
      break;
   }
   default:
      vassert(0);
   }
}

/*------------------------------------------------------------*/
/*--- Name helpers                                         ---*/
/*------------------------------------------------------------*/

/* Obtain an acquire/release atomic-instruction suffix. */
static const HChar* nameAqRlSuffix(UInt aqrl)
{
   switch (aqrl) {
   case 0b00:
      return "";
   case 0b01:
      return ".rl";
   case 0b10:
      return ".aq";
   case 0b11:
      return ".aqrl";
   default:
      vpanic("nameAqRlSuffix(riscv64)");
   }
}

/* Obtain a control/status register name. */
static const HChar* nameCSR(UInt csr)
{
   switch (csr) {
   case 0x001:
      return "fflags";
   case 0x002:
      return "frm";
   case 0x003:
      return "fcsr";
   case 0xc20:
      return "vl";
   case 0xc22:
      return "vlenb";
   default:
      vpanic("nameCSR(riscv64)");
   }
}

/* Obtain a floating-point rounding-mode operand string. */
static const HChar* nameRMOperand(UInt rm)
{
   switch (rm) {
   case 0b000:
      return ", rne";
   case 0b001:
      return ", rtz";
   case 0b010:
      return ", rdn";
   case 0b011:
      return ", rup";
   case 0b100:
      return ", rmm";
   case 0b101:
      return ", <invalid>";
   case 0b110:
      return ", <invalid>";
   case 0b111:
      return ""; /* dyn */
   default:
      vpanic("nameRMOperand(riscv64)");
   }
}

/*------------------------------------------------------------*/
/*--- Disassemble a single instruction                     ---*/
/*------------------------------------------------------------*/

/* A macro to fish bits out of 'insn' which is a local variable to all
   disassembly functions. */
#define INSN(_bMax, _bMin) SLICE_UInt(insn, (_bMax), (_bMin))

static Bool dis_RV64C(/*MB_OUT*/ DisResult* dres,
                      /*OUT*/ IRSB*         irsb,
                      UInt                  insn,
                      Addr                  guest_pc_curr_instr,
                      Bool                  sigill_diag)
{
   vassert(INSN(1, 0) == 0b00 || INSN(1, 0) == 0b01 || INSN(1, 0) == 0b10);

   /* ---- RV64C compressed instruction set, quadrant 0 ----- */

   /* ------------- c.addi4spn rd, nzuimm[9:2] -------------- */
   if (INSN(1, 0) == 0b00 && INSN(15, 13) == 0b000) {
      UInt rd = INSN(4, 2) + 8;
      UInt nzuimm9_2 =
         INSN(10, 7) << 4 | INSN(12, 11) << 2 | INSN(5, 5) << 1 | INSN(6, 6);
      if (nzuimm9_2 == 0) {
         /* Invalid C.ADDI4SPN, fall through. */
      } else {
         ULong uimm = nzuimm9_2 << 2;
         putIReg64(irsb, rd,
                   binop(Iop_Add64, getIReg64(2 /*x2/sp*/), mkU64(uimm)));
         DIP("c.addi4spn %s, %llu\n", nameIReg(rd), uimm);
         return True;
      }
   }

   /* -------------- c.fld rd, uimm[7:3](rs1) --------------- */
   if (INSN(1, 0) == 0b00 && INSN(15, 13) == 0b001) {
      UInt  rd      = INSN(4, 2) + 8;
      UInt  rs1     = INSN(9, 7) + 8;
      UInt  uimm7_3 = INSN(6, 5) << 3 | INSN(12, 10);
      ULong uimm    = uimm7_3 << 3;
      putFReg64(irsb, rd,
                loadLE(Ity_F64, binop(Iop_Add64, getIReg64(rs1), mkU64(uimm))));
      DIP("c.fld %s, %llu(%s)\n", nameFReg(rd), uimm, nameIReg(rs1));
      return True;
   }

   /* --------------- c.lw rd, uimm[6:2](rs1) --------------- */
   if (INSN(1, 0) == 0b00 && INSN(15, 13) == 0b010) {
      UInt  rd      = INSN(4, 2) + 8;
      UInt  rs1     = INSN(9, 7) + 8;
      UInt  uimm6_2 = INSN(5, 5) << 4 | INSN(12, 10) << 1 | INSN(6, 6);
      ULong uimm    = uimm6_2 << 2;
      putIReg64(
         irsb, rd,
         unop(Iop_32Sto64,
              loadLE(Ity_I32, binop(Iop_Add64, getIReg64(rs1), mkU64(uimm)))));
      DIP("c.lw %s, %llu(%s)\n", nameIReg(rd), uimm, nameIReg(rs1));
      return True;
   }

   /* --------------- c.ld rd, uimm[7:3](rs1) --------------- */
   if (INSN(1, 0) == 0b00 && INSN(15, 13) == 0b011) {
      UInt  rd      = INSN(4, 2) + 8;
      UInt  rs1     = INSN(9, 7) + 8;
      UInt  uimm7_3 = INSN(6, 5) << 3 | INSN(12, 10);
      ULong uimm    = uimm7_3 << 3;
      putIReg64(irsb, rd,
                loadLE(Ity_I64, binop(Iop_Add64, getIReg64(rs1), mkU64(uimm))));
      DIP("c.ld %s, %llu(%s)\n", nameIReg(rd), uimm, nameIReg(rs1));
      return True;
   }

   /* -------------- c.fsd rs2, uimm[7:3](rs1) -------------- */
   if (INSN(1, 0) == 0b00 && INSN(15, 13) == 0b101) {
      UInt  rs1     = INSN(9, 7) + 8;
      UInt  rs2     = INSN(4, 2) + 8;
      UInt  uimm7_3 = INSN(6, 5) << 3 | INSN(12, 10);
      ULong uimm    = uimm7_3 << 3;
      storeLE(irsb, binop(Iop_Add64, getIReg64(rs1), mkU64(uimm)),
              getFReg64(rs2));
      DIP("c.fsd %s, %llu(%s)\n", nameFReg(rs2), uimm, nameIReg(rs1));
      return True;
   }

   /* -------------- c.sw rs2, uimm[6:2](rs1) --------------- */
   if (INSN(1, 0) == 0b00 && INSN(15, 13) == 0b110) {
      UInt  rs1     = INSN(9, 7) + 8;
      UInt  rs2     = INSN(4, 2) + 8;
      UInt  uimm6_2 = INSN(5, 5) << 4 | INSN(12, 10) << 1 | INSN(6, 6);
      ULong uimm    = uimm6_2 << 2;
      storeLE(irsb, binop(Iop_Add64, getIReg64(rs1), mkU64(uimm)),
              unop(Iop_64to32, getIReg64(rs2)));
      DIP("c.sw %s, %llu(%s)\n", nameIReg(rs2), uimm, nameIReg(rs1));
      return True;
   }

   /* -------------- c.sd rs2, uimm[7:3](rs1) --------------- */
   if (INSN(1, 0) == 0b00 && INSN(15, 13) == 0b111) {
      UInt  rs1     = INSN(9, 7) + 8;
      UInt  rs2     = INSN(4, 2) + 8;
      UInt  uimm7_3 = INSN(6, 5) << 3 | INSN(12, 10);
      ULong uimm    = uimm7_3 << 3;
      storeLE(irsb, binop(Iop_Add64, getIReg64(rs1), mkU64(uimm)),
              getIReg64(rs2));
      DIP("c.sd %s, %llu(%s)\n", nameIReg(rs2), uimm, nameIReg(rs1));
      return True;
   }

   /* ---- RV64C compressed instruction set, quadrant 1 ----- */

   /* ------------------------ c.nop ------------------------ */
   if (INSN(15, 0) == 0b0000000000000001) {
      DIP("c.nop\n");
      return True;
   }

   /* -------------- c.addi rd_rs1, nzimm[5:0] -------------- */
   if (INSN(1, 0) == 0b01 && INSN(15, 13) == 0b000) {
      UInt rd_rs1   = INSN(11, 7);
      UInt nzimm5_0 = INSN(12, 12) << 5 | INSN(6, 2);
      if (rd_rs1 == 0 || nzimm5_0 == 0) {
         /* Invalid C.ADDI, fall through. */
      } else {
         ULong simm = vex_sx_to_64(nzimm5_0, 6);
         putIReg64(irsb, rd_rs1,
                   binop(Iop_Add64, getIReg64(rd_rs1), mkU64(simm)));
         DIP("c.addi %s, %lld\n", nameIReg(rd_rs1), (Long)simm);
         return True;
      }
   }

   /* -------------- c.addiw rd_rs1, imm[5:0] --------------- */
   if (INSN(1, 0) == 0b01 && INSN(15, 13) == 0b001) {
      UInt rd_rs1 = INSN(11, 7);
      UInt imm5_0 = INSN(12, 12) << 5 | INSN(6, 2);
      if (rd_rs1 == 0) {
         /* Invalid C.ADDIW, fall through. */
      } else {
         UInt simm = (UInt)vex_sx_to_64(imm5_0, 6);
         putIReg32(irsb, rd_rs1,
                   binop(Iop_Add32, getIReg32(rd_rs1), mkU32(simm)));
         DIP("c.addiw %s, %d\n", nameIReg(rd_rs1), (Int)simm);
         return True;
      }
   }

   /* ------------------ c.li rd, imm[5:0] ------------------ */
   if (INSN(1, 0) == 0b01 && INSN(15, 13) == 0b010) {
      UInt rd     = INSN(11, 7);
      UInt imm5_0 = INSN(12, 12) << 5 | INSN(6, 2);
      if (rd == 0) {
         /* Invalid C.LI, fall through. */
      } else {
         ULong simm = vex_sx_to_64(imm5_0, 6);
         putIReg64(irsb, rd, mkU64(simm));
         DIP("c.li %s, %lld\n", nameIReg(rd), (Long)simm);
         return True;
      }
   }

   /* ---------------- c.addi16sp nzimm[9:4] ---------------- */
   if (INSN(1, 0) == 0b01 && INSN(15, 13) == 0b011) {
      UInt rd_rs1   = INSN(11, 7);
      UInt nzimm9_4 = INSN(12, 12) << 5 | INSN(4, 3) << 3 | INSN(5, 5) << 2 |
                      INSN(2, 2) << 1 | INSN(6, 6);
      if (rd_rs1 != 2 || nzimm9_4 == 0) {
         /* Invalid C.ADDI16SP, fall through. */
      } else {
         ULong simm = vex_sx_to_64(nzimm9_4 << 4, 10);
         putIReg64(irsb, rd_rs1,
                   binop(Iop_Add64, getIReg64(rd_rs1), mkU64(simm)));
         DIP("c.addi16sp %lld\n", (Long)simm);
         return True;
      }
   }

   /* --------------- c.lui rd, nzimm[17:12] ---------------- */
   if (INSN(1, 0) == 0b01 && INSN(15, 13) == 0b011) {
      UInt rd         = INSN(11, 7);
      UInt nzimm17_12 = INSN(12, 12) << 5 | INSN(6, 2);
      if (rd == 0 || rd == 2 || nzimm17_12 == 0) {
         /* Invalid C.LUI, fall through. */
      } else {
         putIReg64(irsb, rd, mkU64(vex_sx_to_64(nzimm17_12 << 12, 18)));
         DIP("c.lui %s, 0x%x\n", nameIReg(rd), nzimm17_12);
         return True;
      }
   }

   /* ---------- c.{srli,srai} rd_rs1, nzuimm[5:0] ---------- */
   if (INSN(1, 0) == 0b01 && INSN(11, 11) == 0b0 && INSN(15, 13) == 0b100) {
      Bool is_log    = INSN(10, 10) == 0b0;
      UInt rd_rs1    = INSN(9, 7) + 8;
      UInt nzuimm5_0 = INSN(12, 12) << 5 | INSN(6, 2);
      if (nzuimm5_0 == 0) {
         /* Invalid C.{SRLI,SRAI}, fall through. */
      } else {
         putIReg64(irsb, rd_rs1,
                   binop(is_log ? Iop_Shr64 : Iop_Sar64, getIReg64(rd_rs1),
                         mkU8(nzuimm5_0)));
         DIP("c.%s %s, %u\n", is_log ? "srli" : "srai", nameIReg(rd_rs1),
             nzuimm5_0);
         return True;
      }
   }

   /* --------------- c.andi rd_rs1, imm[5:0] --------------- */
   if (INSN(1, 0) == 0b01 && INSN(11, 10) == 0b10 && INSN(15, 13) == 0b100) {
      UInt rd_rs1 = INSN(9, 7) + 8;
      UInt imm5_0 = INSN(12, 12) << 5 | INSN(6, 2);
      if (rd_rs1 == 0) {
         /* Invalid C.ANDI, fall through. */
      } else {
         ULong simm = vex_sx_to_64(imm5_0, 6);
         putIReg64(irsb, rd_rs1,
                   binop(Iop_And64, getIReg64(rd_rs1), mkU64(simm)));
         DIP("c.andi %s, 0x%llx\n", nameIReg(rd_rs1), simm);
         return True;
      }
   }

   /* ----------- c.{sub,xor,or,and} rd_rs1, rs2 ----------- */
   if (INSN(1, 0) == 0b01 && INSN(15, 10) == 0b100011) {
      UInt         funct2 = INSN(6, 5);
      UInt         rd_rs1 = INSN(9, 7) + 8;
      UInt         rs2    = INSN(4, 2) + 8;
      const HChar* name;
      IROp         op;
      switch (funct2) {
      case 0b00:
         name = "sub";
         op   = Iop_Sub64;
         break;
      case 0b01:
         name = "xor";
         op   = Iop_Xor64;
         break;
      case 0b10:
         name = "or";
         op   = Iop_Or64;
         break;
      case 0b11:
         name = "and";
         op   = Iop_And64;
         break;
      default:
         vassert(0);
      }
      putIReg64(irsb, rd_rs1, binop(op, getIReg64(rd_rs1), getIReg64(rs2)));
      DIP("c.%s %s, %s\n", name, nameIReg(rd_rs1), nameIReg(rs2));
      return True;
   }

   /* -------------- c.{subw,addw} rd_rs1, rs2 -------------- */
   if (INSN(1, 0) == 0b01 && INSN(6, 6) == 0b0 && INSN(15, 10) == 0b100111) {
      Bool is_sub = INSN(5, 5) == 0b0;
      UInt rd_rs1 = INSN(9, 7) + 8;
      UInt rs2    = INSN(4, 2) + 8;
      putIReg32(irsb, rd_rs1,
                binop(is_sub ? Iop_Sub32 : Iop_Add32, getIReg32(rd_rs1),
                      getIReg32(rs2)));
      DIP("c.%s %s, %s\n", is_sub ? "subw" : "addw", nameIReg(rd_rs1),
          nameIReg(rs2));
      return True;
   }

   /* -------------------- c.j imm[11:1] -------------------- */
   if (INSN(1, 0) == 0b01 && INSN(15, 13) == 0b101) {
      UInt imm11_1 = INSN(12, 12) << 10 | INSN(8, 8) << 9 | INSN(10, 9) << 7 |
                     INSN(6, 6) << 6 | INSN(7, 7) << 5 | INSN(2, 2) << 4 |
                     INSN(11, 11) << 3 | INSN(5, 3);
      ULong simm   = vex_sx_to_64(imm11_1 << 1, 12);
      ULong dst_pc = guest_pc_curr_instr + simm;
      putPC(irsb, mkU64(dst_pc));
      dres->whatNext    = Dis_StopHere;
      dres->jk_StopHere = Ijk_Boring;
      DIP("c.j 0x%llx\n", dst_pc);
      return True;
   }

   /* ------------- c.{beqz,bnez} rs1, imm[8:1] ------------- */
   if (INSN(1, 0) == 0b01 && INSN(15, 14) == 0b11) {
      Bool is_eq  = INSN(13, 13) == 0b0;
      UInt rs1    = INSN(9, 7) + 8;
      UInt imm8_1 = INSN(12, 12) << 7 | INSN(6, 5) << 5 | INSN(2, 2) << 4 |
                    INSN(11, 10) << 2 | INSN(4, 3);
      ULong simm   = vex_sx_to_64(imm8_1 << 1, 9);
      ULong dst_pc = guest_pc_curr_instr + simm;
      stmt(irsb, IRStmt_Exit(binop(is_eq ? Iop_CmpEQ64 : Iop_CmpNE64,
                                   getIReg64(rs1), mkU64(0)),
                             Ijk_Boring, IRConst_U64(dst_pc), OFFB_PC));
      putPC(irsb, mkU64(guest_pc_curr_instr + 2));
      dres->whatNext    = Dis_StopHere;
      dres->jk_StopHere = Ijk_Boring;
      DIP("c.%s %s, 0x%llx\n", is_eq ? "beqz" : "bnez", nameIReg(rs1), dst_pc);
      return True;
   }

   /* ---- RV64C compressed instruction set, quadrant 2 ----- */

   /* ------------- c.slli rd_rs1, nzuimm[5:0] -------------- */
   if (INSN(1, 0) == 0b10 && INSN(15, 13) == 0b000) {
      UInt rd_rs1    = INSN(11, 7);
      UInt nzuimm5_0 = INSN(12, 12) << 5 | INSN(6, 2);
      if (rd_rs1 == 0 || nzuimm5_0 == 0) {
         /* Invalid C.SLLI, fall through. */
      } else {
         putIReg64(irsb, rd_rs1,
                   binop(Iop_Shl64, getIReg64(rd_rs1), mkU8(nzuimm5_0)));
         DIP("c.slli %s, %u\n", nameIReg(rd_rs1), nzuimm5_0);
         return True;
      }
   }

   /* -------------- c.fldsp rd, uimm[8:3](x2) -------------- */
   if (INSN(1, 0) == 0b10 && INSN(15, 13) == 0b001) {
      UInt  rd      = INSN(11, 7);
      UInt  rs1     = 2; /* base=x2/sp */
      UInt  uimm8_3 = INSN(4, 2) << 3 | INSN(12, 12) << 2 | INSN(6, 5);
      ULong uimm    = uimm8_3 << 3;
      putFReg64(irsb, rd,
                loadLE(Ity_F64, binop(Iop_Add64, getIReg64(rs1), mkU64(uimm))));
      DIP("c.fldsp %s, %llu(%s)\n", nameFReg(rd), uimm, nameIReg(rs1));
      return True;
   }

   /* -------------- c.lwsp rd, uimm[7:2](x2) --------------- */
   if (INSN(1, 0) == 0b10 && INSN(15, 13) == 0b010) {
      UInt rd      = INSN(11, 7);
      UInt rs1     = 2; /* base=x2/sp */
      UInt uimm7_2 = INSN(3, 2) << 4 | INSN(12, 12) << 3 | INSN(6, 4);
      if (rd == 0) {
         /* Invalid C.LWSP, fall through. */
      } else {
         ULong uimm = uimm7_2 << 2;
         putIReg64(irsb, rd,
                   unop(Iop_32Sto64,
                        loadLE(Ity_I32,
                               binop(Iop_Add64, getIReg64(rs1), mkU64(uimm)))));
         DIP("c.lwsp %s, %llu(%s)\n", nameIReg(rd), uimm, nameIReg(rs1));
         return True;
      }
   }

   /* -------------- c.ldsp rd, uimm[8:3](x2) --------------- */
   if (INSN(1, 0) == 0b10 && INSN(15, 13) == 0b011) {
      UInt rd      = INSN(11, 7);
      UInt rs1     = 2; /* base=x2/sp */
      UInt uimm8_3 = INSN(4, 2) << 3 | INSN(12, 12) << 2 | INSN(6, 5);
      if (rd == 0) {
         /* Invalid C.LDSP, fall through. */
      } else {
         ULong uimm = uimm8_3 << 3;
         putIReg64(
            irsb, rd,
            loadLE(Ity_I64, binop(Iop_Add64, getIReg64(rs1), mkU64(uimm))));
         DIP("c.ldsp %s, %llu(%s)\n", nameIReg(rd), uimm, nameIReg(rs1));
         return True;
      }
   }

   /* ---------------------- c.jr rs1 ----------------------- */
   if (INSN(1, 0) == 0b10 && INSN(15, 12) == 0b1000) {
      UInt rs1 = INSN(11, 7);
      UInt rs2 = INSN(6, 2);
      if (rs1 == 0 || rs2 != 0) {
         /* Invalid C.JR, fall through. */
      } else {
         putPC(irsb, getIReg64(rs1));
         dres->whatNext = Dis_StopHere;
         if (rs1 == 1 /*x1/ra*/) {
            dres->jk_StopHere = Ijk_Ret;
            DIP("c.ret\n");
         } else {
            dres->jk_StopHere = Ijk_Boring;
            DIP("c.jr %s\n", nameIReg(rs1));
         }
         return True;
      }
   }

   /* -------------------- c.mv rd, rs2 --------------------- */
   if (INSN(1, 0) == 0b10 && INSN(15, 12) == 0b1000) {
      UInt rd  = INSN(11, 7);
      UInt rs2 = INSN(6, 2);
      if (rd == 0 || rs2 == 0) {
         /* Invalid C.MV, fall through. */
      } else {
         putIReg64(irsb, rd, getIReg64(rs2));
         DIP("c.mv %s, %s\n", nameIReg(rd), nameIReg(rs2));
         return True;
      }
   }

   /* --------------------- c.ebreak ------------------------ */
   if (INSN(15, 0) == 0b1001000000000010) {
      putPC(irsb, mkU64(guest_pc_curr_instr + 2));
      dres->whatNext    = Dis_StopHere;
      dres->jk_StopHere = Ijk_SigTRAP;
      DIP("c.ebreak\n");
      return True;
   }

   /* --------------------- c.jalr rs1 ---------------------- */
   if (INSN(1, 0) == 0b10 && INSN(15, 12) == 0b1001) {
      UInt rs1 = INSN(11, 7);
      UInt rs2 = INSN(6, 2);
      if (rs1 == 0 || rs2 != 0) {
         /* Invalid C.JALR, fall through. */
      } else {
         putIReg64(irsb, 1 /*x1/ra*/, mkU64(guest_pc_curr_instr + 2));
         putPC(irsb, getIReg64(rs1));
         dres->whatNext    = Dis_StopHere;
         dres->jk_StopHere = Ijk_Call;
         DIP("c.jalr %s\n", nameIReg(rs1));
         return True;
      }
   }

   /* ------------------ c.add rd_rs1, rs2 ------------------ */
   if (INSN(1, 0) == 0b10 && INSN(15, 12) == 0b1001) {
      UInt rd_rs1 = INSN(11, 7);
      UInt rs2    = INSN(6, 2);
      if (rd_rs1 == 0 || rs2 == 0) {
         /* Invalid C.ADD, fall through. */
      } else {
         putIReg64(irsb, rd_rs1,
                   binop(Iop_Add64, getIReg64(rd_rs1), getIReg64(rs2)));
         DIP("c.add %s, %s\n", nameIReg(rd_rs1), nameIReg(rs2));
         return True;
      }
   }

   /* ------------- c.fsdsp rs2, uimm[8:3](x2) -------------- */
   if (INSN(1, 0) == 0b10 && INSN(15, 13) == 0b101) {
      UInt  rs1     = 2; /* base=x2/sp */
      UInt  rs2     = INSN(6, 2);
      UInt  uimm8_3 = INSN(9, 7) << 3 | INSN(12, 10);
      ULong uimm    = uimm8_3 << 3;
      storeLE(irsb, binop(Iop_Add64, getIReg64(rs1), mkU64(uimm)),
              getFReg64(rs2));
      DIP("c.fsdsp %s, %llu(%s)\n", nameFReg(rs2), uimm, nameIReg(rs1));
      return True;
   }

   /* -------------- c.swsp rs2, uimm[7:2](x2) -------------- */
   if (INSN(1, 0) == 0b10 && INSN(15, 13) == 0b110) {
      UInt  rs1     = 2; /* base=x2/sp */
      UInt  rs2     = INSN(6, 2);
      UInt  uimm7_2 = INSN(8, 7) << 4 | INSN(12, 9);
      ULong uimm    = uimm7_2 << 2;
      storeLE(irsb, binop(Iop_Add64, getIReg64(rs1), mkU64(uimm)),
              unop(Iop_64to32, getIReg64(rs2)));
      DIP("c.swsp %s, %llu(%s)\n", nameIReg(rs2), uimm, nameIReg(rs1));
      return True;
   }

   /* -------------- c.sdsp rs2, uimm[8:3](x2) -------------- */
   if (INSN(1, 0) == 0b10 && INSN(15, 13) == 0b111) {
      UInt  rs1     = 2; /* base=x2/sp */
      UInt  rs2     = INSN(6, 2);
      UInt  uimm8_3 = INSN(9, 7) << 3 | INSN(12, 10);
      ULong uimm    = uimm8_3 << 3;
      storeLE(irsb, binop(Iop_Add64, getIReg64(rs1), mkU64(uimm)),
              getIReg64(rs2));
      DIP("c.sdsp %s, %llu(%s)\n", nameIReg(rs2), uimm, nameIReg(rs1));
      return True;
   }

   if (sigill_diag)
      vex_printf("RISCV64 front end: compressed\n");
   return False;
}

static Bool dis_RV64I(/*MB_OUT*/ DisResult* dres,
                      /*OUT*/ IRSB*         irsb,
                      UInt                  insn,
                      Addr                  guest_pc_curr_instr)
{
   /* ------------- RV64I base instruction set -------------- */

   /* ----------------- lui rd, imm[31:12] ------------------ */
   if (INSN(6, 0) == 0b0110111) {
      UInt rd       = INSN(11, 7);
      UInt imm31_12 = INSN(31, 12);
      if (rd != 0)
         putIReg64(irsb, rd, mkU64(vex_sx_to_64(imm31_12 << 12, 32)));
      DIP("lui %s, 0x%x\n", nameIReg(rd), imm31_12);
      return True;
   }

   /* ---------------- auipc rd, imm[31:12] ----------------- */
   if (INSN(6, 0) == 0b0010111) {
      UInt rd       = INSN(11, 7);
      UInt imm31_12 = INSN(31, 12);
      if (rd != 0)
         putIReg64(
            irsb, rd,
            mkU64(guest_pc_curr_instr + vex_sx_to_64(imm31_12 << 12, 32)));
      DIP("auipc %s, 0x%x\n", nameIReg(rd), imm31_12);
      return True;
   }

   /* ------------------ jal rd, imm[20:1] ------------------ */
   if (INSN(6, 0) == 0b1101111) {
      UInt rd      = INSN(11, 7);
      UInt imm20_1 = INSN(31, 31) << 19 | INSN(19, 12) << 11 |
                     INSN(20, 20) << 10 | INSN(30, 21);
      ULong simm   = vex_sx_to_64(imm20_1 << 1, 21);
      ULong dst_pc = guest_pc_curr_instr + simm;
      if (rd != 0)
         putIReg64(irsb, rd, mkU64(guest_pc_curr_instr + 4));
      putPC(irsb, mkU64(dst_pc));
      dres->whatNext = Dis_StopHere;
      if (rd != 0) {
         dres->jk_StopHere = Ijk_Call;
         DIP("jal %s, 0x%llx\n", nameIReg(rd), dst_pc);
      } else {
         dres->jk_StopHere = Ijk_Boring;
         DIP("j 0x%llx\n", dst_pc);
      }
      return True;
   }

   /* --------------- jalr rd, imm[11:0](rs1) --------------- */
   if (INSN(6, 0) == 0b1100111 && INSN(14, 12) == 0b000) {
      UInt   rd      = INSN(11, 7);
      UInt   rs1     = INSN(19, 15);
      UInt   imm11_0 = INSN(31, 20);
      ULong  simm    = vex_sx_to_64(imm11_0, 12);
      IRTemp dst_pc  = newTemp(irsb, Ity_I64);
      assign(irsb, dst_pc, binop(Iop_Add64, getIReg64(rs1), mkU64(simm)));
      if (rd != 0)
         putIReg64(irsb, rd, mkU64(guest_pc_curr_instr + 4));
      putPC(irsb, mkexpr(dst_pc));
      dres->whatNext = Dis_StopHere;
      if (rd == 0) {
         if (rs1 == 1 /*x1/ra*/ && simm == 0) {
            dres->jk_StopHere = Ijk_Ret;
            DIP("ret\n");
         } else {
            dres->jk_StopHere = Ijk_Boring;
            DIP("jr %lld(%s)\n", (Long)simm, nameIReg(rs1));
         }
      } else {
         dres->jk_StopHere = Ijk_Call;
         DIP("jalr %s, %lld(%s)\n", nameIReg(rd), (Long)simm, nameIReg(rs1));
      }
      return True;
   }

   /* ------------ {beq,bne} rs1, rs2, imm[12:1] ------------ */
   /* ------------ {blt,bge} rs1, rs2, imm[12:1] ------------ */
   /* ----------- {bltu,bgeu} rs1, rs2, imm[12:1] ----------- */
   if (INSN(6, 0) == 0b1100011) {
      UInt funct3  = INSN(14, 12);
      UInt rs1     = INSN(19, 15);
      UInt rs2     = INSN(24, 20);
      UInt imm12_1 = INSN(31, 31) << 11 | INSN(7, 7) << 10 | INSN(30, 25) << 4 |
                     INSN(11, 8);
      if (funct3 == 0b010 || funct3 == 0b011) {
         /* Invalid B<x>, fall through. */
      } else {
         ULong        simm   = vex_sx_to_64(imm12_1 << 1, 13);
         ULong        dst_pc = guest_pc_curr_instr + simm;
         const HChar* name;
         IRExpr*      cond;
         switch (funct3) {
         case 0b000:
            name = "beq";
            cond = binop(Iop_CmpEQ64, getIReg64(rs1), getIReg64(rs2));
            break;
         case 0b001:
            name = "bne";
            cond = binop(Iop_CmpNE64, getIReg64(rs1), getIReg64(rs2));
            break;
         case 0b100:
            name = "blt";
            cond = binop(Iop_CmpLT64S, getIReg64(rs1), getIReg64(rs2));
            break;
         case 0b101:
            name = "bge";
            cond = binop(Iop_CmpLE64S, getIReg64(rs2), getIReg64(rs1));
            break;
         case 0b110:
            name = "bltu";
            cond = binop(Iop_CmpLT64U, getIReg64(rs1), getIReg64(rs2));
            break;
         case 0b111:
            name = "bgeu";
            cond = binop(Iop_CmpLE64U, getIReg64(rs2), getIReg64(rs1));
            break;
         default:
            vassert(0);
         }
         stmt(irsb,
              IRStmt_Exit(cond, Ijk_Boring, IRConst_U64(dst_pc), OFFB_PC));
         putPC(irsb, mkU64(guest_pc_curr_instr + 4));
         dres->whatNext    = Dis_StopHere;
         dres->jk_StopHere = Ijk_Boring;
         DIP("%s %s, %s, 0x%llx\n", name, nameIReg(rs1), nameIReg(rs2), dst_pc);
         return True;
      }
   }

   /* ---------- {lb,lh,lw,ld} rd, imm[11:0](rs1) ----------- */
   /* ---------- {lbu,lhu,lwu} rd, imm[11:0](rs1) ----------- */
   if (INSN(6, 0) == 0b0000011) {
      UInt funct3  = INSN(14, 12);
      UInt rd      = INSN(11, 7);
      UInt rs1     = INSN(19, 15);
      UInt imm11_0 = INSN(31, 20);
      if (funct3 == 0b111) {
         /* Invalid L<x>, fall through. */
      } else {
         ULong simm = vex_sx_to_64(imm11_0, 12);
         if (rd != 0) {
            IRExpr* ea = binop(Iop_Add64, getIReg64(rs1), mkU64(simm));
            IRExpr* expr;
            switch (funct3) {
            case 0b000:
               expr = unop(Iop_8Sto64, loadLE(Ity_I8, ea));
               break;
            case 0b001:
               expr = unop(Iop_16Sto64, loadLE(Ity_I16, ea));
               break;
            case 0b010:
               expr = unop(Iop_32Sto64, loadLE(Ity_I32, ea));
               break;
            case 0b011:
               expr = loadLE(Ity_I64, ea);
               break;
            case 0b100:
               expr = unop(Iop_8Uto64, loadLE(Ity_I8, ea));
               break;
            case 0b101:
               expr = unop(Iop_16Uto64, loadLE(Ity_I16, ea));
               break;
            case 0b110:
               expr = unop(Iop_32Uto64, loadLE(Ity_I32, ea));
               break;
            default:
               vassert(0);
            }
            putIReg64(irsb, rd, expr);
         }
         const HChar* name;
         switch (funct3) {
         case 0b000:
            name = "lb";
            break;
         case 0b001:
            name = "lh";
            break;
         case 0b010:
            name = "lw";
            break;
         case 0b011:
            name = "ld";
            break;
         case 0b100:
            name = "lbu";
            break;
         case 0b101:
            name = "lhu";
            break;
         case 0b110:
            name = "lwu";
            break;
         default:
            vassert(0);
         }
         DIP("%s %s, %lld(%s)\n", name, nameIReg(rd), (Long)simm,
             nameIReg(rs1));
         return True;
      }
   }

   /* ---------- {sb,sh,sw,sd} rs2, imm[11:0](rs1) ---------- */
   if (INSN(6, 0) == 0b0100011) {
      UInt funct3  = INSN(14, 12);
      UInt rs1     = INSN(19, 15);
      UInt rs2     = INSN(24, 20);
      UInt imm11_0 = INSN(31, 25) << 5 | INSN(11, 7);
      if (funct3 == 0b100 || funct3 == 0b101 || funct3 == 0b110 ||
          funct3 == 0b111) {
         /* Invalid S<x>, fall through. */
      } else {
         ULong        simm = vex_sx_to_64(imm11_0, 12);
         IRExpr*      ea   = binop(Iop_Add64, getIReg64(rs1), mkU64(simm));
         const HChar* name;
         IRExpr*      expr;
         switch (funct3) {
         case 0b000:
            name = "sb";
            expr = unop(Iop_64to8, getIReg64(rs2));
            break;
         case 0b001:
            name = "sh";
            expr = unop(Iop_64to16, getIReg64(rs2));
            break;
         case 0b010:
            name = "sw";
            expr = unop(Iop_64to32, getIReg64(rs2));
            break;
         case 0b011:
            name = "sd";
            expr = getIReg64(rs2);
            break;
         default:
            vassert(0);
         }
         storeLE(irsb, ea, expr);
         DIP("%s %s, %lld(%s)\n", name, nameIReg(rs2), (Long)simm,
             nameIReg(rs1));
         return True;
      }
   }

   /* -------- {addi,slti,sltiu} rd, rs1, imm[11:0] --------- */
   /* --------- {xori,ori,andi} rd, rs1, imm[11:0] ---------- */
   if (INSN(6, 0) == 0b0010011) {
      UInt funct3  = INSN(14, 12);
      UInt rd      = INSN(11, 7);
      UInt rs1     = INSN(19, 15);
      UInt imm11_0 = INSN(31, 20);
      if (funct3 == 0b001 || funct3 == 0b101) {
         /* Invalid <x>I, fall through. */
      } else {
         ULong simm = vex_sx_to_64(imm11_0, 12);
         if (rd != 0) {
            IRExpr* expr;
            switch (funct3) {
            case 0b000:
               expr = binop(Iop_Add64, getIReg64(rs1), mkU64(simm));
               break;
            case 0b010:
               expr = unop(Iop_1Uto64,
                           binop(Iop_CmpLT64S, getIReg64(rs1), mkU64(simm)));
               break;
            case 0b011:
               /* Note that the comparison itself is unsigned but the immediate
                  is sign-extended. */
               expr = unop(Iop_1Uto64,
                           binop(Iop_CmpLT64U, getIReg64(rs1), mkU64(simm)));
               break;
            case 0b100:
               expr = binop(Iop_Xor64, getIReg64(rs1), mkU64(simm));
               break;
            case 0b110:
               expr = binop(Iop_Or64, getIReg64(rs1), mkU64(simm));
               break;
            case 0b111:
               expr = binop(Iop_And64, getIReg64(rs1), mkU64(simm));
               break;
            default:
               vassert(0);
            }
            putIReg64(irsb, rd, expr);
         }
         const HChar* name;
         switch (funct3) {
         case 0b000:
            name = "addi";
            break;
         case 0b010:
            name = "slti";
            break;
         case 0b011:
            name = "sltiu";
            break;
         case 0b100:
            name = "xori";
            break;
         case 0b110:
            name = "ori";
            break;
         case 0b111:
            name = "andi";
            break;
         default:
            vassert(0);
         }
         DIP("%s %s, %s, %lld\n", name, nameIReg(rd), nameIReg(rs1),
             (Long)simm);
         return True;
      }
   }

   /* --------------- slli rd, rs1, uimm[5:0] --------------- */
   if (INSN(6, 0) == 0b0010011 && INSN(14, 12) == 0b001 &&
       INSN(31, 26) == 0b000000) {
      UInt rd      = INSN(11, 7);
      UInt rs1     = INSN(19, 15);
      UInt uimm5_0 = INSN(25, 20);
      if (rd != 0)
         putIReg64(irsb, rd, binop(Iop_Shl64, getIReg64(rs1), mkU8(uimm5_0)));
      DIP("slli %s, %s, %u\n", nameIReg(rd), nameIReg(rs1), uimm5_0);
      return True;
   }

   /* ----------- {srli,srai} rd, rs1, uimm[5:0] ----------=- */
   if (INSN(6, 0) == 0b0010011 && INSN(14, 12) == 0b101 &&
       INSN(29, 26) == 0b0000 && INSN(31, 31) == 0b0) {
      Bool is_log  = INSN(30, 30) == 0b0;
      UInt rd      = INSN(11, 7);
      UInt rs1     = INSN(19, 15);
      UInt uimm5_0 = INSN(25, 20);
      if (rd != 0)
         putIReg64(irsb, rd,
                   binop(is_log ? Iop_Shr64 : Iop_Sar64, getIReg64(rs1),
                         mkU8(uimm5_0)));
      DIP("%s %s, %s, %u\n", is_log ? "srli" : "srai", nameIReg(rd),
          nameIReg(rs1), uimm5_0);
      return True;
   }

   /* --------------- {add,sub} rd, rs1, rs2 ---------------- */
   /* ------------- {sll,srl,sra} rd, rs1, rs2 -------------- */
   /* --------------- {slt,sltu} rd, rs1, rs2 --------------- */
   /* -------------- {xor,or,and} rd, rs1, rs2 -------------- */
   if (INSN(6, 0) == 0b0110011 && INSN(29, 25) == 0b00000 &&
       INSN(31, 31) == 0b0) {
      UInt funct3  = INSN(14, 12);
      Bool is_base = INSN(30, 30) == 0b0;
      UInt rd      = INSN(11, 7);
      UInt rs1     = INSN(19, 15);
      UInt rs2     = INSN(24, 20);
      if (!is_base && funct3 != 0b000 && funct3 != 0b101) {
         /* Invalid <x>, fall through. */
      } else {
         if (rd != 0) {
            IRExpr* expr;
            switch (funct3) {
            case 0b000: /* sll */
               expr = binop(is_base ? Iop_Add64 : Iop_Sub64, getIReg64(rs1),
                            getIReg64(rs2));
               break;
            case 0b001:
               expr = binop(Iop_Shl64, getIReg64(rs1),
                            unop(Iop_64to8, getIReg64(rs2)));
               break;
            case 0b010:
               expr = unop(Iop_1Uto64,
                           binop(Iop_CmpLT64S, getIReg64(rs1), getIReg64(rs2)));
               break;
            case 0b011:
               expr = unop(Iop_1Uto64,
                           binop(Iop_CmpLT64U, getIReg64(rs1), getIReg64(rs2)));
               break;
            case 0b100:
               expr = binop(Iop_Xor64, getIReg64(rs1), getIReg64(rs2));
               break;
            case 0b101:
               expr = binop(is_base ? Iop_Shr64 : Iop_Sar64, getIReg64(rs1),
                            unop(Iop_64to8, getIReg64(rs2)));
               break;
            case 0b110:
               expr = binop(Iop_Or64, getIReg64(rs1), getIReg64(rs2));
               break;
            case 0b111:
               expr = binop(Iop_And64, getIReg64(rs1), getIReg64(rs2));
               break;
            default:
               vassert(0);
            }
            putIReg64(irsb, rd, expr);
         }
         const HChar* name;
         switch (funct3) {
         case 0b000:
            name = is_base ? "add" : "sub";
            break;
         case 0b001:
            name = "sll";
            break;
         case 0b010:
            name = "slt";
            break;
         case 0b011:
            name = "sltu";
            break;
         case 0b100:
            name = "xor";
            break;
         case 0b101:
            name = is_base ? "srl" : "sra";
            break;
         case 0b110:
            name = "or";
            break;
         case 0b111:
            name = "and";
            break;
         default:
            vassert(0);
         }
         DIP("%s %s, %s, %s\n", name, nameIReg(rd), nameIReg(rs1),
             nameIReg(rs2));
         return True;
      }
   }

   /* ------------------------ fence ------------------------ */
   if (INSN(19, 0) == 0b00000000000000001111 && INSN(31, 28) == 0b0000) {
      UInt succ = INSN(23, 20);
      UInt pred = INSN(27, 24);
      stmt(irsb, IRStmt_MBE(Imbe_Fence));
      if (pred == 0b1111 && succ == 0b1111)
         DIP("fence\n");
      else
         DIP("fence %s%s%s%s,%s%s%s%s\n", (pred & 0x8) ? "i" : "",
             (pred & 0x4) ? "o" : "", (pred & 0x2) ? "r" : "",
             (pred & 0x1) ? "w" : "", (succ & 0x8) ? "i" : "",
             (succ & 0x4) ? "o" : "", (succ & 0x2) ? "r" : "",
             (succ & 0x1) ? "w" : "");
      return True;
   }

   /* ------------------------ ecall ------------------------ */
   if (INSN(31, 0) == 0b00000000000000000000000001110011) {
      putPC(irsb, mkU64(guest_pc_curr_instr + 4));
      dres->whatNext    = Dis_StopHere;
      dres->jk_StopHere = Ijk_Sys_syscall;
      DIP("ecall\n");
      return True;
   }

   /* ------------------------ ebreak ------------------------ */
   if (INSN(31, 0) == 0b00000000000100000000000001110011) {
      putPC(irsb, mkU64(guest_pc_curr_instr + 4));
      dres->whatNext    = Dis_StopHere;
      dres->jk_StopHere = Ijk_SigTRAP;
      DIP("ebreak\n");
      return True;
   }

   /* -------------- addiw rd, rs1, imm[11:0] --------------- */
   if (INSN(6, 0) == 0b0011011 && INSN(14, 12) == 0b000) {
      UInt rd      = INSN(11, 7);
      UInt rs1     = INSN(19, 15);
      UInt imm11_0 = INSN(31, 20);
      UInt simm    = (UInt)vex_sx_to_64(imm11_0, 12);
      if (rd != 0)
         putIReg32(irsb, rd, binop(Iop_Add32, getIReg32(rs1), mkU32(simm)));
      DIP("addiw %s, %s, %d\n", nameIReg(rd), nameIReg(rs1), (Int)simm);
      return True;
   }

   /* -------------- slliw rd, rs1, uimm[4:0] --------------- */
   if (INSN(6, 0) == 0b0011011 && INSN(14, 12) == 0b001 &&
       INSN(31, 25) == 0b0000000) {
      UInt rd      = INSN(11, 7);
      UInt rs1     = INSN(19, 15);
      UInt uimm4_0 = INSN(24, 20);
      if (rd != 0)
         putIReg32(irsb, rd, binop(Iop_Shl32, getIReg32(rs1), mkU8(uimm4_0)));
      DIP("slliw %s, %s, %u\n", nameIReg(rd), nameIReg(rs1), uimm4_0);
      return True;
   }

   /* ---------- {srliw,sraiw} rd, rs1, uimm[4:0] ----------- */
   if (INSN(6, 0) == 0b0011011 && INSN(14, 12) == 0b101 &&
       INSN(29, 25) == 0b00000 && INSN(31, 31) == 0b0) {
      Bool is_log  = INSN(30, 30) == 0b0;
      UInt rd      = INSN(11, 7);
      UInt rs1     = INSN(19, 15);
      UInt uimm4_0 = INSN(24, 20);
      if (rd != 0)
         putIReg32(irsb, rd,
                   binop(is_log ? Iop_Shr32 : Iop_Sar32, getIReg32(rs1),
                         mkU8(uimm4_0)));
      DIP("%s %s, %s, %u\n", is_log ? "srliw" : "sraiw", nameIReg(rd),
          nameIReg(rs1), uimm4_0);
      return True;
   }

   /* -------------- {addw,subw} rd, rs1, rs2 --------------- */
   if (INSN(6, 0) == 0b0111011 && INSN(14, 12) == 0b000 &&
       INSN(29, 25) == 0b00000 && INSN(31, 31) == 0b0) {
      Bool is_add = INSN(30, 30) == 0b0;
      UInt rd     = INSN(11, 7);
      UInt rs1    = INSN(19, 15);
      UInt rs2    = INSN(24, 20);
      if (rd != 0)
         putIReg32(irsb, rd,
                   binop(is_add ? Iop_Add32 : Iop_Sub32, getIReg32(rs1),
                         getIReg32(rs2)));
      DIP("%s %s, %s, %s\n", is_add ? "addw" : "subw", nameIReg(rd),
          nameIReg(rs1), nameIReg(rs2));
      return True;
   }

   /* ------------------ sllw rd, rs1, rs2 ------------------ */
   if (INSN(6, 0) == 0b0111011 && INSN(14, 12) == 0b001 &&
       INSN(31, 25) == 0b0000000) {
      UInt rd  = INSN(11, 7);
      UInt rs1 = INSN(19, 15);
      UInt rs2 = INSN(24, 20);
      if (rd != 0)
         putIReg32(
            irsb, rd,
            binop(Iop_Shl32, getIReg32(rs1), unop(Iop_64to8, getIReg64(rs2))));
      DIP("sllw %s, %s, %s\n", nameIReg(rd), nameIReg(rs1), nameIReg(rs2));
      return True;
   }

   /* -------------- {srlw,sraw} rd, rs1, rs2 --------------- */
   if (INSN(6, 0) == 0b0111011 && INSN(14, 12) == 0b101 &&
       INSN(29, 25) == 0b00000 && INSN(31, 31) == 0b0) {
      Bool is_log = INSN(30, 30) == 0b0;
      UInt rd     = INSN(11, 7);
      UInt rs1    = INSN(19, 15);
      UInt rs2    = INSN(24, 20);
      if (rd != 0)
         putIReg32(irsb, rd,
                   binop(is_log ? Iop_Shr32 : Iop_Sar32, getIReg32(rs1),
                         unop(Iop_64to8, getIReg64(rs2))));
      DIP("%s %s, %s, %s\n", is_log ? "srlw" : "sraw", nameIReg(rd),
          nameIReg(rs1), nameIReg(rs2));
      return True;
   }

   return False;
}

static Bool dis_RV64M(/*MB_OUT*/ DisResult* dres,
                      /*OUT*/ IRSB*         irsb,
                      UInt                  insn)
{
   /* -------------- RV64M standard extension --------------- */

   /* -------- {mul,mulh,mulhsu,mulhu} rd, rs1, rs2 --------- */
   /* --------------- {div,divu} rd, rs1, rs2 --------------- */
   /* --------------- {rem,remu} rd, rs1, rs2 --------------- */
   if (INSN(6, 0) == 0b0110011 && INSN(31, 25) == 0b0000001) {
      UInt rd     = INSN(11, 7);
      UInt funct3 = INSN(14, 12);
      UInt rs1    = INSN(19, 15);
      UInt rs2    = INSN(24, 20);
      if (funct3 == 0b010) {
         /* Invalid {MUL,DIV,REM}<x>, fall through. */
      } else if (funct3 == 0b010) {
         /* MULHSU, not currently handled, fall through. */
      } else {
         if (rd != 0) {
            IRExpr* expr;
            switch (funct3) {
            case 0b000:
               expr = binop(Iop_Mul64, getIReg64(rs1), getIReg64(rs2));
               break;
            case 0b001:
               expr = unop(Iop_128HIto64,
                           binop(Iop_MullS64, getIReg64(rs1), getIReg64(rs2)));
               break;
            case 0b011:
               expr = unop(Iop_128HIto64,
                           binop(Iop_MullU64, getIReg64(rs1), getIReg64(rs2)));
               break;
            case 0b100:
               expr = binop(Iop_DivS64, getIReg64(rs1), getIReg64(rs2));
               break;
            case 0b101:
               expr = binop(Iop_DivU64, getIReg64(rs1), getIReg64(rs2));
               break;
            case 0b110:
               expr =
                  unop(Iop_128HIto64, binop(Iop_DivModS64to64, getIReg64(rs1),
                                            getIReg64(rs2)));
               break;
            case 0b111:
               expr =
                  unop(Iop_128HIto64, binop(Iop_DivModU64to64, getIReg64(rs1),
                                            getIReg64(rs2)));
               break;
            default:
               vassert(0);
            }
            putIReg64(irsb, rd, expr);
         }
         const HChar* name;
         switch (funct3) {
         case 0b000:
            name = "mul";
            break;
         case 0b001:
            name = "mulh";
            break;
         case 0b011:
            name = "mulhu";
            break;
         case 0b100:
            name = "div";
            break;
         case 0b101:
            name = "divu";
            break;
         case 0b110:
            name = "rem";
            break;
         case 0b111:
            name = "remu";
            break;
         default:
            vassert(0);
         }
         DIP("%s %s, %s, %s\n", name, nameIReg(rd), nameIReg(rs1),
             nameIReg(rs2));
         return True;
      }
   }

   /* ------------------ mulw rd, rs1, rs2 ------------------ */
   /* -------------- {divw,divuw} rd, rs1, rs2 -------------- */
   /* -------------- {remw,remuw} rd, rs1, rs2 -------------- */
   if (INSN(6, 0) == 0b0111011 && INSN(31, 25) == 0b0000001) {
      UInt rd     = INSN(11, 7);
      UInt funct3 = INSN(14, 12);
      UInt rs1    = INSN(19, 15);
      UInt rs2    = INSN(24, 20);
      if (funct3 == 0b001 || funct3 == 0b010 || funct3 == 0b011) {
         /* Invalid {MUL,DIV,REM}<x>W, fall through. */
      } else {
         if (rd != 0) {
            IRExpr* expr;
            switch (funct3) {
            case 0b000:
               expr = binop(Iop_Mul32, getIReg32(rs1), getIReg32(rs2));
               break;
            case 0b100:
               expr = binop(Iop_DivS32, getIReg32(rs1), getIReg32(rs2));
               break;
            case 0b101:
               expr = binop(Iop_DivU32, getIReg32(rs1), getIReg32(rs2));
               break;
            case 0b110:
               expr = unop(Iop_64HIto32, binop(Iop_DivModS32to32,
                                               getIReg32(rs1), getIReg32(rs2)));
               break;
            case 0b111:
               expr = unop(Iop_64HIto32, binop(Iop_DivModU32to32,
                                               getIReg32(rs1), getIReg32(rs2)));
               break;
            default:
               vassert(0);
            }
            putIReg32(irsb, rd, expr);
         }
         const HChar* name;
         switch (funct3) {
         case 0b000:
            name = "mulw";
            break;
         case 0b100:
            name = "divw";
            break;
         case 0b101:
            name = "divuw";
            break;
         case 0b110:
            name = "remw";
            break;
         case 0b111:
            name = "remuw";
            break;
         default:
            vassert(0);
         }
         DIP("%s %s, %s, %s\n", name, nameIReg(rd), nameIReg(rs1),
             nameIReg(rs2));
         return True;
      }
   }

   return False;
}

static Bool dis_RV64A(/*MB_OUT*/ DisResult* dres,
                      /*OUT*/ IRSB*         irsb,
                      UInt                  insn,
                      Addr                  guest_pc_curr_instr,
                      const VexAbiInfo*     abiinfo)
{
   /* -------------- RV64A standard extension --------------- */

   /* ----------------- lr.{w,d} rd, (rs1) ------------------ */
   if (INSN(6, 0) == 0b0101111 && INSN(14, 13) == 0b01 &&
       INSN(24, 20) == 0b00000 && INSN(31, 27) == 0b00010) {
      UInt rd    = INSN(11, 7);
      Bool is_32 = INSN(12, 12) == 0b0;
      UInt rs1   = INSN(19, 15);
      UInt aqrl  = INSN(26, 25);

      if (aqrl & 0x1)
         stmt(irsb, IRStmt_MBE(Imbe_Fence));

      IRType ty = is_32 ? Ity_I32 : Ity_I64;
      if (abiinfo->guest__use_fallback_LLSC) {
         /* Get address of the load. */
         IRTemp ea = newTemp(irsb, Ity_I64);
         assign(irsb, ea, getIReg64(rs1));

         /* Load the value. */
         IRTemp res = newTemp(irsb, Ity_I64);
         assign(irsb, res, widenSto64(ty, loadLE(ty, mkexpr(ea))));

         /* Set up the LLSC fallback data. */
         stmt(irsb, IRStmt_Put(OFFB_LLSC_DATA, mkexpr(res)));
         stmt(irsb, IRStmt_Put(OFFB_LLSC_ADDR, mkexpr(ea)));
         stmt(irsb, IRStmt_Put(OFFB_LLSC_SIZE, mkU64(4)));

         /* Write the result to the destination register. */
         if (rd != 0)
            putIReg64(irsb, rd, mkexpr(res));
      } else {
         /* TODO Rework the non-fallback mode by recognizing common LR+SC
            sequences and simulating them as one. */
         IRTemp res = newTemp(irsb, ty);
         stmt(irsb, IRStmt_LLSC(Iend_LE, res, getIReg64(rs1), NULL /*LL*/));
         if (rd != 0)
            putIReg64(irsb, rd, widenSto64(ty, mkexpr(res)));
      }

      if (aqrl & 0x2)
         stmt(irsb, IRStmt_MBE(Imbe_Fence));

      DIP("lr.%s%s %s, (%s)%s\n", is_32 ? "w" : "d", nameAqRlSuffix(aqrl),
          nameIReg(rd), nameIReg(rs1),
          abiinfo->guest__use_fallback_LLSC ? " (fallback implementation)"
                                            : "");
      return True;
   }

   /* --------------- sc.{w,d} rd, rs2, (rs1) --------------- */
   if (INSN(6, 0) == 0b0101111 && INSN(14, 13) == 0b01 &&
       INSN(31, 27) == 0b00011) {
      UInt rd    = INSN(11, 7);
      Bool is_32 = INSN(12, 12) == 0b0;
      UInt rs1   = INSN(19, 15);
      UInt rs2   = INSN(24, 20);
      UInt aqrl  = INSN(26, 25);

      if (aqrl & 0x1)
         stmt(irsb, IRStmt_MBE(Imbe_Fence));

      IRType ty = is_32 ? Ity_I32 : Ity_I64;
      if (abiinfo->guest__use_fallback_LLSC) {
         /* Get address of the load. */
         IRTemp ea = newTemp(irsb, Ity_I64);
         assign(irsb, ea, getIReg64(rs1));

         /* Get the continuation address. */
         IRConst* nia = IRConst_U64(guest_pc_curr_instr + 4);

         /* Mark the SC initially as failed. */
         if (rd != 0)
            putIReg64(irsb, rd, mkU64(1));

         /* Set that no transaction is in progress. */
         IRTemp size = newTemp(irsb, Ity_I64);
         assign(irsb, size, IRExpr_Get(OFFB_LLSC_SIZE, Ity_I64));
         stmt(irsb,
              IRStmt_Put(OFFB_LLSC_SIZE, mkU64(0) /* "no transaction" */));

         /* Fail if no or wrong-size transaction. */
         stmt(irsb, IRStmt_Exit(binop(Iop_CmpNE64, mkexpr(size), mkU64(4)),
                                Ijk_Boring, nia, OFFB_PC));

         /* Fail if the address doesn't match the LL address. */
         stmt(irsb, IRStmt_Exit(binop(Iop_CmpNE64, mkexpr(ea),
                                      IRExpr_Get(OFFB_LLSC_ADDR, Ity_I64)),
                                Ijk_Boring, nia, OFFB_PC));

         /* Fail if the data doesn't match the LL data. */
         IRTemp data = newTemp(irsb, Ity_I64);
         assign(irsb, data, IRExpr_Get(OFFB_LLSC_DATA, Ity_I64));
         stmt(irsb, IRStmt_Exit(binop(Iop_CmpNE64,
                                      widenSto64(ty, loadLE(ty, mkexpr(ea))),
                                      mkexpr(data)),
                                Ijk_Boring, nia, OFFB_PC));

         /* Try to CAS the new value in. */
         IRTemp old  = newTemp(irsb, ty);
         IRTemp expd = newTemp(irsb, ty);
         assign(irsb, expd, narrowFrom64(ty, mkexpr(data)));
         stmt(irsb, IRStmt_CAS(mkIRCAS(
                       /*oldHi*/ IRTemp_INVALID, old, Iend_LE, mkexpr(ea),
                       /*expdHi*/ NULL, mkexpr(expd),
                       /*dataHi*/ NULL, narrowFrom64(ty, getIReg64(rs2)))));

         /* Fail if the CAS failed (old != expd). */
         stmt(irsb, IRStmt_Exit(binop(is_32 ? Iop_CmpNE32 : Iop_CmpNE64,
                                      mkexpr(old), mkexpr(expd)),
                                Ijk_Boring, nia, OFFB_PC));

         /* Otherwise mark the operation as successful. */
         if (rd != 0)
            putIReg64(irsb, rd, mkU64(0));
      } else {
         IRTemp res = newTemp(irsb, Ity_I1);
         stmt(irsb, IRStmt_LLSC(Iend_LE, res, getIReg64(rs1),
                                narrowFrom64(ty, getIReg64(rs2))));
         /* IR semantics: res is 1 if store succeeds, 0 if it fails. Need to set
            rd to 1 on failure, 0 on success. */
         if (rd != 0)
            putIReg64(
               irsb, rd,
               binop(Iop_Xor64, unop(Iop_1Uto64, mkexpr(res)), mkU64(1)));
      }

      if (aqrl & 0x2)
         stmt(irsb, IRStmt_MBE(Imbe_Fence));

      DIP("sc.%s%s %s, %s, (%s)%s\n", is_32 ? "w" : "d", nameAqRlSuffix(aqrl),
          nameIReg(rd), nameIReg(rs2), nameIReg(rs1),
          abiinfo->guest__use_fallback_LLSC ? " (fallback implementation)"
                                            : "");
      return True;
   }

   /* --------- amo{swap,add}.{w,d} rd, rs2, (rs1) ---------- */
   /* -------- amo{xor,and,or}.{w,d} rd, rs2, (rs1) --------- */
   /* ---------- amo{min,max}.{w,d} rd, rs2, (rs1) ---------- */
   /* --------- amo{minu,maxu}.{w,d} rd, rs2, (rs1) --------- */
   if (INSN(6, 0) == 0b0101111 && INSN(14, 13) == 0b01) {
      UInt rd     = INSN(11, 7);
      Bool is_32  = INSN(12, 12) == 0b0;
      UInt rs1    = INSN(19, 15);
      UInt rs2    = INSN(24, 20);
      UInt aqrl   = INSN(26, 25);
      UInt funct5 = INSN(31, 27);
      if ((funct5 & 0b00010) || funct5 == 0b00101 || funct5 == 0b01001 ||
          funct5 == 0b01101 || funct5 == 0b10001 || funct5 == 0b10101 ||
          funct5 == 0b11001 || funct5 == 0b11101) {
         /* Invalid AMO<x>, fall through. */
      } else {
         if (aqrl & 0x1)
            stmt(irsb, IRStmt_MBE(Imbe_Fence));

         IRTemp addr = newTemp(irsb, Ity_I64);
         assign(irsb, addr, getIReg64(rs1));

         IRType ty   = is_32 ? Ity_I32 : Ity_I64;
         IRTemp orig = newTemp(irsb, ty);
         assign(irsb, orig, loadLE(ty, mkexpr(addr)));
         IRExpr* lhs = mkexpr(orig);
         IRExpr* rhs = narrowFrom64(ty, getIReg64(rs2));

         /* Perform the operation. */
         const HChar* name;
         IRExpr*      res;
         switch (funct5) {
         case 0b00001:
            name = "amoswap";
            res  = rhs;
            break;
         case 0b00000:
            name = "amoadd";
            res  = binop(is_32 ? Iop_Add32 : Iop_Add64, lhs, rhs);
            break;
         case 0b00100:
            name = "amoxor";
            res  = binop(is_32 ? Iop_Xor32 : Iop_Xor64, lhs, rhs);
            break;
         case 0b01100:
            name = "amoand";
            res  = binop(is_32 ? Iop_And32 : Iop_And64, lhs, rhs);
            break;
         case 0b01000:
            name = "amoor";
            res  = binop(is_32 ? Iop_Or32 : Iop_Or64, lhs, rhs);
            break;
         case 0b10000:
            name = "amomin";
            res  = IRExpr_ITE(
                binop(is_32 ? Iop_CmpLT32S : Iop_CmpLT64S, lhs, rhs), lhs, rhs);
            break;
         case 0b10100:
            name = "amomax";
            res  = IRExpr_ITE(
                binop(is_32 ? Iop_CmpLT32S : Iop_CmpLT64S, lhs, rhs), rhs, lhs);
            break;
         case 0b11000:
            name = "amominu";
            res  = IRExpr_ITE(
                binop(is_32 ? Iop_CmpLT32U : Iop_CmpLT64U, lhs, rhs), lhs, rhs);
            break;
         case 0b11100:
            name = "amomaxu";
            res  = IRExpr_ITE(
                binop(is_32 ? Iop_CmpLT32U : Iop_CmpLT64U, lhs, rhs), rhs, lhs);
            break;
         default:
            vassert(0);
         }

         /* Store the result back if the original value remains unchanged in
            memory. */
         IRTemp old = newTemp(irsb, ty);
         stmt(irsb, IRStmt_CAS(mkIRCAS(/*oldHi*/ IRTemp_INVALID, old, Iend_LE,
                                       mkexpr(addr),
                                       /*expdHi*/ NULL, mkexpr(orig),
                                       /*dataHi*/ NULL, res)));

         if (aqrl & 0x2)
            stmt(irsb, IRStmt_MBE(Imbe_Fence));

         /* Retry if the CAS failed (i.e. when old != orig). */
         stmt(irsb, IRStmt_Exit(binop(is_32 ? Iop_CasCmpNE32 : Iop_CasCmpNE64,
                                      mkexpr(old), mkexpr(orig)),
                                Ijk_Boring, IRConst_U64(guest_pc_curr_instr),
                                OFFB_PC));
         /* Otherwise we succeeded. */
         if (rd != 0)
            putIReg64(irsb, rd, widenSto64(ty, mkexpr(old)));

         DIP("%s.%s%s %s, %s, (%s)\n", name, is_32 ? "w" : "d",
             nameAqRlSuffix(aqrl), nameIReg(rd), nameIReg(rs2), nameIReg(rs1));
         return True;
      }
   }

   return False;
}

static Bool dis_RV64F(/*MB_OUT*/ DisResult* dres,
                      /*OUT*/ IRSB*         irsb,
                      UInt                  insn)
{
   /* -------------- RV64F standard extension --------------- */

   /* --------------- flw rd, imm[11:0](rs1) ---------------- */
   if (INSN(6, 0) == 0b0000111 && INSN(14, 12) == 0b010) {
      UInt  rd      = INSN(11, 7);
      UInt  rs1     = INSN(19, 15);
      UInt  imm11_0 = INSN(31, 20);
      ULong simm    = vex_sx_to_64(imm11_0, 12);
      putFReg32(irsb, rd,
                loadLE(Ity_F32, binop(Iop_Add64, getIReg64(rs1), mkU64(simm))));
      DIP("flw %s, %lld(%s)\n", nameFReg(rd), (Long)simm, nameIReg(rs1));
      return True;
   }

   /* --------------- fsw rs2, imm[11:0](rs1) --------------- */
   if (INSN(6, 0) == 0b0100111 && INSN(14, 12) == 0b010) {
      UInt  rs1     = INSN(19, 15);
      UInt  rs2     = INSN(24, 20);
      UInt  imm11_0 = INSN(31, 25) << 5 | INSN(11, 7);
      ULong simm    = vex_sx_to_64(imm11_0, 12);
      storeLE(irsb, binop(Iop_Add64, getIReg64(rs1), mkU64(simm)),
              getFReg32(rs2));
      DIP("fsw %s, %lld(%s)\n", nameFReg(rs2), (Long)simm, nameIReg(rs1));
      return True;
   }

   /* -------- f{madd,msub}.s rd, rs1, rs2, rs3, rm --------- */
   /* ------- f{nmsub,nmadd}.s rd, rs1, rs2, rs3, rm -------- */
   if (INSN(1, 0) == 0b11 && INSN(6, 4) == 0b100 && INSN(26, 25) == 0b00) {
      UInt   opcode = INSN(6, 0);
      UInt   rd     = INSN(11, 7);
      UInt   rm     = INSN(14, 12);
      UInt   rs1    = INSN(19, 15);
      UInt   rs2    = INSN(24, 20);
      UInt   rs3    = INSN(31, 27);
      IRTemp rm_RISCV, rm_IR;
      mk_get_rounding_mode(irsb, &rm_RISCV, &rm_IR, rm);
      const HChar* name;
      IRTemp       a1 = newTemp(irsb, Ity_F32);
      IRTemp       a2 = newTemp(irsb, Ity_F32);
      IRTemp       a3 = newTemp(irsb, Ity_F32);
      switch (opcode) {
      case 0b1000011:
         name = "fmadd";
         assign(irsb, a1, getFReg32(rs1));
         assign(irsb, a2, getFReg32(rs2));
         assign(irsb, a3, getFReg32(rs3));
         break;
      case 0b1000111:
         name = "fmsub";
         assign(irsb, a1, getFReg32(rs1));
         assign(irsb, a2, getFReg32(rs2));
         assign(irsb, a3, unop(Iop_NegF32, getFReg32(rs3)));
         break;
      case 0b1001011:
         name = "fnmsub";
         assign(irsb, a1, unop(Iop_NegF32, getFReg32(rs1)));
         assign(irsb, a2, getFReg32(rs2));
         assign(irsb, a3, getFReg32(rs3));
         break;
      case 0b1001111:
         name = "fnmadd";
         assign(irsb, a1, unop(Iop_NegF32, getFReg32(rs1)));
         assign(irsb, a2, getFReg32(rs2));
         assign(irsb, a3, unop(Iop_NegF32, getFReg32(rs3)));
         break;
      default:
         vassert(0);
      }
      putFReg32(
         irsb, rd,
         qop(Iop_MAddF32, mkexpr(rm_IR), mkexpr(a1), mkexpr(a2), mkexpr(a3)));
      accumulateFFLAGS(
         irsb, mkIRExprCCall(Ity_I32, 0 /*regparms*/,
                             "riscv64g_calculate_fflags_fmadd_s",
                             riscv64g_calculate_fflags_fmadd_s,
                             mkIRExprVec_4(mkexpr(a1), mkexpr(a2), mkexpr(a3),
                                           mkexpr(rm_RISCV))));
      DIP("%s.s %s, %s, %s, %s%s\n", name, nameFReg(rd), nameFReg(rs1),
          nameFReg(rs2), nameFReg(rs3), nameRMOperand(rm));
      return True;
   }

   /* ------------ f{add,sub}.s rd, rs1, rs2, rm ------------ */
   /* ------------ f{mul,div}.s rd, rs1, rs2, rm ------------ */
   if (INSN(6, 0) == 0b1010011 && INSN(26, 25) == 0b00 &&
       INSN(31, 29) == 0b000) {
      UInt   rd     = INSN(11, 7);
      UInt   rm     = INSN(14, 12);
      UInt   rs1    = INSN(19, 15);
      UInt   rs2    = INSN(24, 20);
      UInt   funct7 = INSN(31, 25);
      IRTemp rm_RISCV, rm_IR;
      mk_get_rounding_mode(irsb, &rm_RISCV, &rm_IR, rm);
      const HChar* name;
      IROp         op;
      IRTemp       a1 = newTemp(irsb, Ity_F32);
      IRTemp       a2 = newTemp(irsb, Ity_F32);
      const HChar* helper_name;
      void*        helper_addr;
      switch (funct7) {
      case 0b0000000:
         name = "fadd";
         op   = Iop_AddF32;
         assign(irsb, a1, getFReg32(rs1));
         assign(irsb, a2, getFReg32(rs2));
         helper_name = "riscv64g_calculate_fflags_fadd_s";
         helper_addr = riscv64g_calculate_fflags_fadd_s;
         break;
      case 0b0000100:
         name = "fsub";
         op   = Iop_AddF32;
         assign(irsb, a1, getFReg32(rs1));
         assign(irsb, a2, unop(Iop_NegF32, getFReg32(rs2)));
         helper_name = "riscv64g_calculate_fflags_fadd_s";
         helper_addr = riscv64g_calculate_fflags_fadd_s;
         break;
      case 0b0001000:
         name = "fmul";
         op   = Iop_MulF32;
         assign(irsb, a1, getFReg32(rs1));
         assign(irsb, a2, getFReg32(rs2));
         helper_name = "riscv64g_calculate_fflags_fmul_s";
         helper_addr = riscv64g_calculate_fflags_fmul_s;
         break;
      case 0b0001100:
         name = "fdiv";
         op   = Iop_DivF32;
         assign(irsb, a1, getFReg32(rs1));
         assign(irsb, a2, getFReg32(rs2));
         helper_name = "riscv64g_calculate_fflags_fdiv_s";
         helper_addr = riscv64g_calculate_fflags_fdiv_s;
         break;
      default:
         vassert(0);
      }
      putFReg32(irsb, rd, triop(op, mkexpr(rm_IR), mkexpr(a1), mkexpr(a2)));
      accumulateFFLAGS(irsb, mkIRExprCCall(Ity_I32, 0 /*regparms*/, helper_name,
                                           helper_addr,
                                           mkIRExprVec_3(mkexpr(a1), mkexpr(a2),
                                                         mkexpr(rm_RISCV))));
      DIP("%s.s %s, %s, %s%s\n", name, nameFReg(rd), nameFReg(rs1),
          nameFReg(rs2), nameRMOperand(rm));
      return True;
   }

   /* ----------------- fsqrt.s rd, rs1, rm ----------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(24, 20) == 0b00000 &&
       INSN(31, 25) == 0b0101100) {
      UInt   rd  = INSN(11, 7);
      UInt   rm  = INSN(14, 12);
      UInt   rs1 = INSN(19, 15);
      IRTemp rm_RISCV, rm_IR;
      mk_get_rounding_mode(irsb, &rm_RISCV, &rm_IR, rm);
      IRTemp a1 = newTemp(irsb, Ity_F32);
      assign(irsb, a1, getFReg32(rs1));
      putFReg32(irsb, rd, binop(Iop_SqrtF32, mkexpr(rm_IR), mkexpr(a1)));
      accumulateFFLAGS(
         irsb, mkIRExprCCall(Ity_I32, 0 /*regparms*/,
                             "riscv64g_calculate_fflags_fsqrt_s",
                             riscv64g_calculate_fflags_fsqrt_s,
                             mkIRExprVec_2(mkexpr(a1), mkexpr(rm_RISCV))));
      DIP("fsqrt.s %s, %s%s\n", nameFReg(rd), nameFReg(rs1), nameRMOperand(rm));
      return True;
   }

   /* ---------------- fsgnj.s rd, rs1, rs2 ----------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(14, 12) == 0b000 &&
       INSN(31, 25) == 0b0010000) {
      UInt rd  = INSN(11, 7);
      UInt rs1 = INSN(19, 15);
      UInt rs2 = INSN(24, 20);
      if (rs1 == rs2) {
         putFReg32(irsb, rd, getFReg32(rs1));
         DIP("fmv.s %s, %s\n", nameFReg(rd), nameIReg(rs1));
      } else {
         putFReg32(
            irsb, rd,
            unop(Iop_ReinterpI32asF32,
                 binop(
                    Iop_Or32,
                    binop(Iop_And32, unop(Iop_ReinterpF32asI32, getFReg32(rs1)),
                          mkU32(0x7fffffff)),
                    binop(Iop_And32, unop(Iop_ReinterpF32asI32, getFReg32(rs2)),
                          mkU32(0x80000000)))));
         DIP("fsgnj.s %s, %s, %s\n", nameFReg(rd), nameIReg(rs1),
             nameIReg(rs2));
      }
      return True;
   }

   /* ---------------- fsgnjn.s rd, rs1, rs2 ---------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(14, 12) == 0b001 &&
       INSN(31, 25) == 0b0010000) {
      UInt rd  = INSN(11, 7);
      UInt rs1 = INSN(19, 15);
      UInt rs2 = INSN(24, 20);
      if (rs1 == rs2) {
         putFReg32(irsb, rd, unop(Iop_NegF32, getFReg32(rs1)));
         DIP("fneg.s %s, %s\n", nameFReg(rd), nameIReg(rs1));
      } else {
         putFReg32(irsb, rd,
                   unop(Iop_ReinterpI32asF32,
                        binop(Iop_Or32,
                              binop(Iop_And32,
                                    unop(Iop_ReinterpF32asI32, getFReg32(rs1)),
                                    mkU32(0x7fffffff)),
                              binop(Iop_And32,
                                    unop(Iop_ReinterpF32asI32,
                                         unop(Iop_NegF32, getFReg32(rs2))),
                                    mkU32(0x80000000)))));
         DIP("fsgnjn.s %s, %s, %s\n", nameFReg(rd), nameIReg(rs1),
             nameIReg(rs2));
      }
      return True;
   }

   /* ---------------- fsgnjx.s rd, rs1, rs2 ---------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(14, 12) == 0b010 &&
       INSN(31, 25) == 0b0010000) {
      UInt rd  = INSN(11, 7);
      UInt rs1 = INSN(19, 15);
      UInt rs2 = INSN(24, 20);
      if (rs1 == rs2) {
         putFReg32(irsb, rd, unop(Iop_AbsF32, getFReg32(rs1)));
         DIP("fabs.s %s, %s\n", nameFReg(rd), nameIReg(rs1));
      } else {
         putFReg32(
            irsb, rd,
            unop(Iop_ReinterpI32asF32,
                 binop(Iop_Xor32, unop(Iop_ReinterpF32asI32, getFReg32(rs1)),
                       binop(Iop_And32,
                             unop(Iop_ReinterpF32asI32, getFReg32(rs2)),
                             mkU32(0x80000000)))));
         DIP("fsgnjx.s %s, %s, %s\n", nameFReg(rd), nameIReg(rs1),
             nameIReg(rs2));
      }
      return True;
   }

   /* -------------- f{min,max}.s rd, rs1, rs2 -------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(31, 25) == 0b0010100) {
      UInt rd  = INSN(11, 7);
      UInt rm  = INSN(14, 12);
      UInt rs1 = INSN(19, 15);
      UInt rs2 = INSN(24, 20);
      if (rm != 0b000 && rm != 0b001) {
         /* Invalid F{MIN,MAX}.S, fall through. */
      } else {
         const HChar* name;
         IROp         op;
         const HChar* helper_name;
         void*        helper_addr;
         switch (rm) {
         case 0b000:
            name        = "fmin";
            op          = Iop_MinNumF32;
            helper_name = "riscv64g_calculate_fflags_fmin_s";
            helper_addr = riscv64g_calculate_fflags_fmin_s;
            break;
         case 0b001:
            name        = "fmax";
            op          = Iop_MaxNumF32;
            helper_name = "riscv64g_calculate_fflags_fmax_s";
            helper_addr = riscv64g_calculate_fflags_fmax_s;
            break;
         default:
            vassert(0);
         }
         IRTemp a1 = newTemp(irsb, Ity_F32);
         IRTemp a2 = newTemp(irsb, Ity_F32);
         assign(irsb, a1, getFReg32(rs1));
         assign(irsb, a2, getFReg32(rs2));
         putFReg32(irsb, rd, binop(op, mkexpr(a1), mkexpr(a2)));
         accumulateFFLAGS(irsb,
                          mkIRExprCCall(Ity_I32, 0 /*regparms*/, helper_name,
                                        helper_addr,
                                        mkIRExprVec_2(mkexpr(a1), mkexpr(a2))));
         DIP("%s.s %s, %s, %s\n", name, nameFReg(rd), nameFReg(rs1),
             nameFReg(rs2));
         return True;
      }
   }

   /* -------------- fcvt.{w,wu}.s rd, rs1, rm -------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(24, 21) == 0b0000 &&
       INSN(31, 25) == 0b1100000) {
      UInt   rd        = INSN(11, 7);
      UInt   rm        = INSN(14, 12);
      UInt   rs1       = INSN(19, 15);
      Bool   is_signed = INSN(20, 20) == 0b0;
      IRTemp rm_RISCV, rm_IR;
      mk_get_rounding_mode(irsb, &rm_RISCV, &rm_IR, rm);
      IRTemp a1 = newTemp(irsb, Ity_F32);
      assign(irsb, a1, getFReg32(rs1));
      if (rd != 0)
         putIReg32(irsb, rd,
                   binop(is_signed ? Iop_F32toI32S : Iop_F32toI32U,
                         mkexpr(rm_IR), mkexpr(a1)));
      accumulateFFLAGS(
         irsb, mkIRExprCCall(Ity_I32, 0 /*regparms*/,
                             is_signed ? "riscv64g_calculate_fflags_fcvt_w_s"
                                       : "riscv64g_calculate_fflags_fcvt_wu_s",
                             is_signed ? riscv64g_calculate_fflags_fcvt_w_s
                                       : riscv64g_calculate_fflags_fcvt_wu_s,
                             mkIRExprVec_2(mkexpr(a1), mkexpr(rm_RISCV))));
      DIP("fcvt.w%s.s %s, %s%s\n", is_signed ? "" : "u", nameIReg(rd),
          nameFReg(rs1), nameRMOperand(rm));
      return True;
   }

   /* ------------------- fmv.x.w rd, rs1 ------------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(14, 12) == 0b000 &&
       INSN(24, 20) == 0b00000 && INSN(31, 25) == 0b1110000) {
      UInt rd  = INSN(11, 7);
      UInt rs1 = INSN(19, 15);
      if (rd != 0)
         putIReg32(irsb, rd, unop(Iop_ReinterpF32asI32, getFReg32(rs1)));
      DIP("fmv.x.w %s, %s\n", nameIReg(rd), nameFReg(rs1));
      return True;
   }

   /* ------------- f{eq,lt,le}.s rd, rs1, rs2 -------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(31, 25) == 0b1010000) {
      UInt rd  = INSN(11, 7);
      UInt rm  = INSN(14, 12);
      UInt rs1 = INSN(19, 15);
      UInt rs2 = INSN(24, 20);
      if (rm != 0b010 && rm != 0b001 && rm != 0b000) {
         /* Invalid F{EQ,LT,LE}.S, fall through. */
      } else {
         IRTemp a1 = newTemp(irsb, Ity_F32);
         IRTemp a2 = newTemp(irsb, Ity_F32);
         assign(irsb, a1, getFReg32(rs1));
         assign(irsb, a2, getFReg32(rs2));
         if (rd != 0) {
            IRTemp cmp = newTemp(irsb, Ity_I32);
            assign(irsb, cmp, binop(Iop_CmpF32, mkexpr(a1), mkexpr(a2)));
            IRTemp res = newTemp(irsb, Ity_I1);
            switch (rm) {
            case 0b010:
               assign(irsb, res,
                      binop(Iop_CmpEQ32, mkexpr(cmp), mkU32(Ircr_EQ)));
               break;
            case 0b001:
               assign(irsb, res,
                      binop(Iop_CmpEQ32, mkexpr(cmp), mkU32(Ircr_LT)));
               break;
            case 0b000:
               assign(irsb, res,
                      binop(Iop_Or1,
                            binop(Iop_CmpEQ32, mkexpr(cmp), mkU32(Ircr_LT)),
                            binop(Iop_CmpEQ32, mkexpr(cmp), mkU32(Ircr_EQ))));
               break;
            default:
               vassert(0);
            }
            putIReg64(irsb, rd, unop(Iop_1Uto64, mkexpr(res)));
         }
         const HChar* name;
         const HChar* helper_name;
         void*        helper_addr;
         switch (rm) {
         case 0b010:
            name        = "feq";
            helper_name = "riscv64g_calculate_fflags_feq_s";
            helper_addr = riscv64g_calculate_fflags_feq_s;
            break;
         case 0b001:
            name        = "flt";
            helper_name = "riscv64g_calculate_fflags_flt_s";
            helper_addr = riscv64g_calculate_fflags_flt_s;
            break;
         case 0b000:
            name        = "fle";
            helper_name = "riscv64g_calculate_fflags_fle_s";
            helper_addr = riscv64g_calculate_fflags_fle_s;
            break;
         default:
            vassert(0);
         }
         accumulateFFLAGS(irsb,
                          mkIRExprCCall(Ity_I32, 0 /*regparms*/, helper_name,
                                        helper_addr,
                                        mkIRExprVec_2(mkexpr(a1), mkexpr(a2))));
         DIP("%s.s %s, %s, %s\n", name, nameIReg(rd), nameFReg(rs1),
             nameFReg(rs2));
         return True;
      }
   }

   /* ------------------ fclass.s rd, rs1 ------------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(14, 12) == 0b001 &&
       INSN(24, 20) == 0b00000 && INSN(31, 25) == 0b1110000) {
      UInt rd  = INSN(11, 7);
      UInt rs1 = INSN(19, 15);
      if (rd != 0)
         putIReg64(irsb, rd,
                   mkIRExprCCall(Ity_I64, 0 /*regparms*/,
                                 "riscv64g_calculate_fclass_s",
                                 riscv64g_calculate_fclass_s,
                                 mkIRExprVec_1(getFReg32(rs1))));
      DIP("fclass.s %s, %s\n", nameIReg(rd), nameFReg(rs1));
      return True;
   }

   /* ------------------- fmv.w.x rd, rs1 ------------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(14, 12) == 0b000 &&
       INSN(24, 20) == 0b00000 && INSN(31, 25) == 0b1111000) {
      UInt rd  = INSN(11, 7);
      UInt rs1 = INSN(19, 15);
      putFReg32(irsb, rd, unop(Iop_ReinterpI32asF32, getIReg32(rs1)));
      DIP("fmv.w.x %s, %s\n", nameFReg(rd), nameIReg(rs1));
      return True;
   }

   /* -------------- fcvt.s.{w,wu} rd, rs1, rm -------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(24, 21) == 0b0000 &&
       INSN(31, 25) == 0b1101000) {
      UInt   rd        = INSN(11, 7);
      UInt   rm        = INSN(14, 12);
      UInt   rs1       = INSN(19, 15);
      Bool   is_signed = INSN(20, 20) == 0b0;
      IRTemp rm_RISCV, rm_IR;
      mk_get_rounding_mode(irsb, &rm_RISCV, &rm_IR, rm);
      IRTemp a1 = newTemp(irsb, Ity_I32);
      assign(irsb, a1, getIReg32(rs1));
      putFReg32(irsb, rd,
                binop(is_signed ? Iop_I32StoF32 : Iop_I32UtoF32, mkexpr(rm_IR),
                      mkexpr(a1)));
      accumulateFFLAGS(
         irsb, mkIRExprCCall(Ity_I32, 0 /*regparms*/,
                             is_signed ? "riscv64g_calculate_fflags_fcvt_s_w"
                                       : "riscv64g_calculate_fflags_fcvt_s_wu",
                             is_signed ? riscv64g_calculate_fflags_fcvt_s_w
                                       : riscv64g_calculate_fflags_fcvt_s_wu,
                             mkIRExprVec_2(mkexpr(a1), mkexpr(rm_RISCV))));
      DIP("fcvt.s.w%s %s, %s%s\n", is_signed ? "" : "u", nameFReg(rd),
          nameIReg(rs1), nameRMOperand(rm));
      return True;
   }

   /* -------------- fcvt.{l,lu}.s rd, rs1, rm -------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(24, 21) == 0b0001 &&
       INSN(31, 25) == 0b1100000) {
      UInt   rd        = INSN(11, 7);
      UInt   rm        = INSN(14, 12);
      UInt   rs1       = INSN(19, 15);
      Bool   is_signed = INSN(20, 20) == 0b0;
      IRTemp rm_RISCV, rm_IR;
      mk_get_rounding_mode(irsb, &rm_RISCV, &rm_IR, rm);
      IRTemp a1 = newTemp(irsb, Ity_F32);
      assign(irsb, a1, getFReg32(rs1));
      if (rd != 0)
         putIReg64(irsb, rd,
                   binop(is_signed ? Iop_F32toI64S : Iop_F32toI64U,
                         mkexpr(rm_IR), mkexpr(a1)));
      accumulateFFLAGS(
         irsb, mkIRExprCCall(Ity_I32, 0 /*regparms*/,
                             is_signed ? "riscv64g_calculate_fflags_fcvt_l_s"
                                       : "riscv64g_calculate_fflags_fcvt_lu_s",
                             is_signed ? riscv64g_calculate_fflags_fcvt_l_s
                                       : riscv64g_calculate_fflags_fcvt_lu_s,
                             mkIRExprVec_2(mkexpr(a1), mkexpr(rm_RISCV))));
      DIP("fcvt.l%s.s %s, %s%s\n", is_signed ? "" : "u", nameIReg(rd),
          nameFReg(rs1), nameRMOperand(rm));
      return True;
   }

   /* -------------- fcvt.s.{l,lu} rd, rs1, rm -------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(24, 21) == 0b0001 &&
       INSN(31, 25) == 0b1101000) {
      UInt   rd        = INSN(11, 7);
      UInt   rm        = INSN(14, 12);
      UInt   rs1       = INSN(19, 15);
      Bool   is_signed = INSN(20, 20) == 0b0;
      IRTemp rm_RISCV, rm_IR;
      mk_get_rounding_mode(irsb, &rm_RISCV, &rm_IR, rm);
      IRTemp a1 = newTemp(irsb, Ity_I64);
      assign(irsb, a1, getIReg64(rs1));
      putFReg32(irsb, rd,
                binop(is_signed ? Iop_I64StoF32 : Iop_I64UtoF32, mkexpr(rm_IR),
                      mkexpr(a1)));
      accumulateFFLAGS(
         irsb, mkIRExprCCall(Ity_I32, 0 /*regparms*/,
                             is_signed ? "riscv64g_calculate_fflags_fcvt_s_l"
                                       : "riscv64g_calculate_fflags_fcvt_s_lu",
                             is_signed ? riscv64g_calculate_fflags_fcvt_s_l
                                       : riscv64g_calculate_fflags_fcvt_s_lu,
                             mkIRExprVec_2(mkexpr(a1), mkexpr(rm_RISCV))));
      DIP("fcvt.s.l%s %s, %s%s\n", is_signed ? "" : "u", nameFReg(rd),
          nameIReg(rs1), nameRMOperand(rm));
      return True;
   }

   return False;
}

static Bool dis_RV64D(/*MB_OUT*/ DisResult* dres,
                      /*OUT*/ IRSB*         irsb,
                      UInt                  insn)
{
   /* -------------- RV64D standard extension --------------- */

   /* --------------- fld rd, imm[11:0](rs1) ---------------- */
   if (INSN(6, 0) == 0b0000111 && INSN(14, 12) == 0b011) {
      UInt  rd      = INSN(11, 7);
      UInt  rs1     = INSN(19, 15);
      UInt  imm11_0 = INSN(31, 20);
      ULong simm    = vex_sx_to_64(imm11_0, 12);
      putFReg64(irsb, rd,
                loadLE(Ity_F64, binop(Iop_Add64, getIReg64(rs1), mkU64(simm))));
      DIP("fld %s, %lld(%s)\n", nameFReg(rd), (Long)simm, nameIReg(rs1));
      return True;
   }

   /* --------------- fsd rs2, imm[11:0](rs1) --------------- */
   if (INSN(6, 0) == 0b0100111 && INSN(14, 12) == 0b011) {
      UInt  rs1     = INSN(19, 15);
      UInt  rs2     = INSN(24, 20);
      UInt  imm11_0 = INSN(31, 25) << 5 | INSN(11, 7);
      ULong simm    = vex_sx_to_64(imm11_0, 12);
      storeLE(irsb, binop(Iop_Add64, getIReg64(rs1), mkU64(simm)),
              getFReg64(rs2));
      DIP("fsd %s, %lld(%s)\n", nameFReg(rs2), (Long)simm, nameIReg(rs1));
      return True;
   }

   /* -------- f{madd,msub}.d rd, rs1, rs2, rs3, rm --------- */
   /* ------- f{nmsub,nmadd}.d rd, rs1, rs2, rs3, rm -------- */
   if (INSN(1, 0) == 0b11 && INSN(6, 4) == 0b100 && INSN(26, 25) == 0b01) {
      UInt   opcode = INSN(6, 0);
      UInt   rd     = INSN(11, 7);
      UInt   rm     = INSN(14, 12);
      UInt   rs1    = INSN(19, 15);
      UInt   rs2    = INSN(24, 20);
      UInt   rs3    = INSN(31, 27);
      IRTemp rm_RISCV, rm_IR;
      mk_get_rounding_mode(irsb, &rm_RISCV, &rm_IR, rm);
      const HChar* name;
      IRTemp       a1 = newTemp(irsb, Ity_F64);
      IRTemp       a2 = newTemp(irsb, Ity_F64);
      IRTemp       a3 = newTemp(irsb, Ity_F64);
      switch (opcode) {
      case 0b1000011:
         name = "fmadd";
         assign(irsb, a1, getFReg64(rs1));
         assign(irsb, a2, getFReg64(rs2));
         assign(irsb, a3, getFReg64(rs3));
         break;
      case 0b1000111:
         name = "fmsub";
         assign(irsb, a1, getFReg64(rs1));
         assign(irsb, a2, getFReg64(rs2));
         assign(irsb, a3, unop(Iop_NegF64, getFReg64(rs3)));
         break;
      case 0b1001011:
         name = "fnmsub";
         assign(irsb, a1, unop(Iop_NegF64, getFReg64(rs1)));
         assign(irsb, a2, getFReg64(rs2));
         assign(irsb, a3, getFReg64(rs3));
         break;
      case 0b1001111:
         name = "fnmadd";
         assign(irsb, a1, unop(Iop_NegF64, getFReg64(rs1)));
         assign(irsb, a2, getFReg64(rs2));
         assign(irsb, a3, unop(Iop_NegF64, getFReg64(rs3)));
         break;
      default:
         vassert(0);
      }
      putFReg64(
         irsb, rd,
         qop(Iop_MAddF64, mkexpr(rm_IR), mkexpr(a1), mkexpr(a2), mkexpr(a3)));
      accumulateFFLAGS(
         irsb, mkIRExprCCall(Ity_I32, 0 /*regparms*/,
                             "riscv64g_calculate_fflags_fmadd_d",
                             riscv64g_calculate_fflags_fmadd_d,
                             mkIRExprVec_4(mkexpr(a1), mkexpr(a2), mkexpr(a3),
                                           mkexpr(rm_RISCV))));
      DIP("%s.d %s, %s, %s, %s%s\n", name, nameFReg(rd), nameFReg(rs1),
          nameFReg(rs2), nameFReg(rs3), nameRMOperand(rm));
      return True;
   }

   /* ------------ f{add,sub}.d rd, rs1, rs2, rm ------------ */
   /* ------------ f{mul,div}.d rd, rs1, rs2, rm ------------ */
   if (INSN(6, 0) == 0b1010011 && INSN(26, 25) == 0b01 &&
       INSN(31, 29) == 0b000) {
      UInt   rd     = INSN(11, 7);
      UInt   rm     = INSN(14, 12);
      UInt   rs1    = INSN(19, 15);
      UInt   rs2    = INSN(24, 20);
      UInt   funct7 = INSN(31, 25);
      IRTemp rm_RISCV, rm_IR;
      mk_get_rounding_mode(irsb, &rm_RISCV, &rm_IR, rm);
      const HChar* name;
      IROp         op;
      IRTemp       a1 = newTemp(irsb, Ity_F64);
      IRTemp       a2 = newTemp(irsb, Ity_F64);
      const HChar* helper_name;
      void*        helper_addr;
      switch (funct7) {
      case 0b0000001:
         name = "fadd";
         op   = Iop_AddF64;
         assign(irsb, a1, getFReg64(rs1));
         assign(irsb, a2, getFReg64(rs2));
         helper_name = "riscv64g_calculate_fflags_fadd_d";
         helper_addr = riscv64g_calculate_fflags_fadd_d;
         break;
      case 0b0000101:
         name = "fsub";
         op   = Iop_AddF64;
         assign(irsb, a1, getFReg64(rs1));
         assign(irsb, a2, unop(Iop_NegF64, getFReg64(rs2)));
         helper_name = "riscv64g_calculate_fflags_fadd_d";
         helper_addr = riscv64g_calculate_fflags_fadd_d;
         break;
      case 0b0001001:
         name = "fmul";
         op   = Iop_MulF64;
         assign(irsb, a1, getFReg64(rs1));
         assign(irsb, a2, getFReg64(rs2));
         helper_name = "riscv64g_calculate_fflags_fmul_d";
         helper_addr = riscv64g_calculate_fflags_fmul_d;
         break;
      case 0b0001101:
         name = "fdiv";
         op   = Iop_DivF64;
         assign(irsb, a1, getFReg64(rs1));
         assign(irsb, a2, getFReg64(rs2));
         helper_name = "riscv64g_calculate_fflags_fdiv_d";
         helper_addr = riscv64g_calculate_fflags_fdiv_d;
         break;
      default:
         vassert(0);
      }
      putFReg64(irsb, rd, triop(op, mkexpr(rm_IR), mkexpr(a1), mkexpr(a2)));
      accumulateFFLAGS(irsb, mkIRExprCCall(Ity_I32, 0 /*regparms*/, helper_name,
                                           helper_addr,
                                           mkIRExprVec_3(mkexpr(a1), mkexpr(a2),
                                                         mkexpr(rm_RISCV))));
      DIP("%s.d %s, %s, %s%s\n", name, nameFReg(rd), nameFReg(rs1),
          nameFReg(rs2), nameRMOperand(rm));
      return True;
   }

   /* ----------------- fsqrt.d rd, rs1, rm ----------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(24, 20) == 0b00000 &&
       INSN(31, 25) == 0b0101101) {
      UInt   rd  = INSN(11, 7);
      UInt   rm  = INSN(14, 12);
      UInt   rs1 = INSN(19, 15);
      IRTemp rm_RISCV, rm_IR;
      mk_get_rounding_mode(irsb, &rm_RISCV, &rm_IR, rm);
      IRTemp a1 = newTemp(irsb, Ity_F64);
      assign(irsb, a1, getFReg64(rs1));
      putFReg64(irsb, rd, binop(Iop_SqrtF64, mkexpr(rm_IR), mkexpr(a1)));
      accumulateFFLAGS(
         irsb, mkIRExprCCall(Ity_I32, 0 /*regparms*/,
                             "riscv64g_calculate_fflags_fsqrt_d",
                             riscv64g_calculate_fflags_fsqrt_d,
                             mkIRExprVec_2(mkexpr(a1), mkexpr(rm_RISCV))));
      DIP("fsqrt.d %s, %s%s\n", nameFReg(rd), nameFReg(rs1), nameRMOperand(rm));
      return True;
   }

   /* ---------------- fsgnj.d rd, rs1, rs2 ----------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(14, 12) == 0b000 &&
       INSN(31, 25) == 0b0010001) {
      UInt rd  = INSN(11, 7);
      UInt rs1 = INSN(19, 15);
      UInt rs2 = INSN(24, 20);
      if (rs1 == rs2) {
         putFReg64(irsb, rd, getFReg64(rs1));
         DIP("fmv.d %s, %s\n", nameFReg(rd), nameIReg(rs1));
      } else {
         putFReg64(
            irsb, rd,
            unop(Iop_ReinterpI64asF64,
                 binop(
                    Iop_Or64,
                    binop(Iop_And64, unop(Iop_ReinterpF64asI64, getFReg64(rs1)),
                          mkU64(0x7fffffffffffffff)),
                    binop(Iop_And64, unop(Iop_ReinterpF64asI64, getFReg64(rs2)),
                          mkU64(0x8000000000000000)))));
         DIP("fsgnj.d %s, %s, %s\n", nameFReg(rd), nameIReg(rs1),
             nameIReg(rs2));
      }
      return True;
   }

   /* ---------------- fsgnjn.d rd, rs1, rs2 ---------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(14, 12) == 0b001 &&
       INSN(31, 25) == 0b0010001) {
      UInt rd  = INSN(11, 7);
      UInt rs1 = INSN(19, 15);
      UInt rs2 = INSN(24, 20);
      if (rs1 == rs2) {
         putFReg64(irsb, rd, unop(Iop_NegF64, getFReg64(rs1)));
         DIP("fneg.d %s, %s\n", nameFReg(rd), nameIReg(rs1));
      } else {
         putFReg64(irsb, rd,
                   unop(Iop_ReinterpI64asF64,
                        binop(Iop_Or64,
                              binop(Iop_And64,
                                    unop(Iop_ReinterpF64asI64, getFReg64(rs1)),
                                    mkU64(0x7fffffffffffffff)),
                              binop(Iop_And64,
                                    unop(Iop_ReinterpF64asI64,
                                         unop(Iop_NegF64, getFReg64(rs2))),
                                    mkU64(0x8000000000000000)))));
         DIP("fsgnjn.d %s, %s, %s\n", nameFReg(rd), nameIReg(rs1),
             nameIReg(rs2));
      }
      return True;
   }

   /* ---------------- fsgnjx.d rd, rs1, rs2 ---------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(14, 12) == 0b010 &&
       INSN(31, 25) == 0b0010001) {
      UInt rd  = INSN(11, 7);
      UInt rs1 = INSN(19, 15);
      UInt rs2 = INSN(24, 20);
      if (rs1 == rs2) {
         putFReg64(irsb, rd, unop(Iop_AbsF64, getFReg64(rs1)));
         DIP("fabs.d %s, %s\n", nameFReg(rd), nameIReg(rs1));
      } else {
         putFReg64(
            irsb, rd,
            unop(Iop_ReinterpI64asF64,
                 binop(Iop_Xor64, unop(Iop_ReinterpF64asI64, getFReg64(rs1)),
                       binop(Iop_And64,
                             unop(Iop_ReinterpF64asI64, getFReg64(rs2)),
                             mkU64(0x8000000000000000)))));
         DIP("fsgnjx.d %s, %s, %s\n", nameFReg(rd), nameIReg(rs1),
             nameIReg(rs2));
      }
      return True;
   }

   /* -------------- f{min,max}.d rd, rs1, rs2 -------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(31, 25) == 0b0010101) {
      UInt rd  = INSN(11, 7);
      UInt rm  = INSN(14, 12);
      UInt rs1 = INSN(19, 15);
      UInt rs2 = INSN(24, 20);
      if (rm != 0b000 && rm != 0b001) {
         /* Invalid F{MIN,MAX}.D, fall through. */
      } else {
         const HChar* name;
         IROp         op;
         const HChar* helper_name;
         void*        helper_addr;
         switch (rm) {
         case 0b000:
            name        = "fmin";
            op          = Iop_MinNumF64;
            helper_name = "riscv64g_calculate_fflags_fmin_d";
            helper_addr = riscv64g_calculate_fflags_fmin_d;
            break;
         case 0b001:
            name        = "fmax";
            op          = Iop_MaxNumF64;
            helper_name = "riscv64g_calculate_fflags_fmax_d";
            helper_addr = riscv64g_calculate_fflags_fmax_d;
            break;
         default:
            vassert(0);
         }
         IRTemp a1 = newTemp(irsb, Ity_F64);
         IRTemp a2 = newTemp(irsb, Ity_F64);
         assign(irsb, a1, getFReg64(rs1));
         assign(irsb, a2, getFReg64(rs2));
         putFReg64(irsb, rd, binop(op, mkexpr(a1), mkexpr(a2)));
         accumulateFFLAGS(irsb,
                          mkIRExprCCall(Ity_I32, 0 /*regparms*/, helper_name,
                                        helper_addr,
                                        mkIRExprVec_2(mkexpr(a1), mkexpr(a2))));
         DIP("%s.d %s, %s, %s\n", name, nameFReg(rd), nameFReg(rs1),
             nameFReg(rs2));
         return True;
      }
   }

   /* ---------------- fcvt.s.d rd, rs1, rm ----------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(24, 20) == 0b00001 &&
       INSN(31, 25) == 0b0100000) {
      UInt   rd  = INSN(11, 7);
      UInt   rm  = INSN(14, 12);
      UInt   rs1 = INSN(19, 15);
      IRTemp rm_RISCV, rm_IR;
      mk_get_rounding_mode(irsb, &rm_RISCV, &rm_IR, rm);
      IRTemp a1 = newTemp(irsb, Ity_F64);
      assign(irsb, a1, getFReg64(rs1));
      putFReg32(irsb, rd, binop(Iop_F64toF32, mkexpr(rm_IR), mkexpr(a1)));
      accumulateFFLAGS(
         irsb, mkIRExprCCall(Ity_I32, 0 /*regparms*/,
                             "riscv64g_calculate_fflags_fcvt_s_d",
                             riscv64g_calculate_fflags_fcvt_s_d,
                             mkIRExprVec_2(mkexpr(a1), mkexpr(rm_RISCV))));
      DIP("fcvt.s.d %s, %s%s\n", nameFReg(rd), nameFReg(rs1),
          nameRMOperand(rm));
      return True;
   }

   /* ---------------- fcvt.d.s rd, rs1, rm ----------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(24, 20) == 0b00000 &&
       INSN(31, 25) == 0b0100001) {
      UInt rd  = INSN(11, 7);
      UInt rm  = INSN(14, 12); /* Ignored as the result is always exact. */
      UInt rs1 = INSN(19, 15);
      putFReg64(irsb, rd, unop(Iop_F32toF64, getFReg32(rs1)));
      DIP("fcvt.d.s %s, %s%s\n", nameFReg(rd), nameFReg(rs1),
          nameRMOperand(rm));
      return True;
   }

   /* ------------- f{eq,lt,le}.d rd, rs1, rs2 -------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(31, 25) == 0b1010001) {
      UInt rd  = INSN(11, 7);
      UInt rm  = INSN(14, 12);
      UInt rs1 = INSN(19, 15);
      UInt rs2 = INSN(24, 20);
      if (rm != 0b010 && rm != 0b001 && rm != 0b000) {
         /* Invalid F{EQ,LT,LE}.D, fall through. */
      } else {
         IRTemp a1 = newTemp(irsb, Ity_F64);
         IRTemp a2 = newTemp(irsb, Ity_F64);
         assign(irsb, a1, getFReg64(rs1));
         assign(irsb, a2, getFReg64(rs2));
         if (rd != 0) {
            IRTemp cmp = newTemp(irsb, Ity_I32);
            assign(irsb, cmp, binop(Iop_CmpF64, mkexpr(a1), mkexpr(a2)));
            IRTemp res = newTemp(irsb, Ity_I1);
            switch (rm) {
            case 0b010:
               assign(irsb, res,
                      binop(Iop_CmpEQ32, mkexpr(cmp), mkU32(Ircr_EQ)));
               break;
            case 0b001:
               assign(irsb, res,
                      binop(Iop_CmpEQ32, mkexpr(cmp), mkU32(Ircr_LT)));
               break;
            case 0b000:
               assign(irsb, res,
                      binop(Iop_Or1,
                            binop(Iop_CmpEQ32, mkexpr(cmp), mkU32(Ircr_LT)),
                            binop(Iop_CmpEQ32, mkexpr(cmp), mkU32(Ircr_EQ))));
               break;
            default:
               vassert(0);
            }
            putIReg64(irsb, rd, unop(Iop_1Uto64, mkexpr(res)));
         }
         const HChar* name;
         const HChar* helper_name;
         void*        helper_addr;
         switch (rm) {
         case 0b010:
            name        = "feq";
            helper_name = "riscv64g_calculate_fflags_feq_d";
            helper_addr = riscv64g_calculate_fflags_feq_d;
            break;
         case 0b001:
            name        = "flt";
            helper_name = "riscv64g_calculate_fflags_flt_d";
            helper_addr = riscv64g_calculate_fflags_flt_d;
            break;
         case 0b000:
            name        = "fle";
            helper_name = "riscv64g_calculate_fflags_fle_d";
            helper_addr = riscv64g_calculate_fflags_fle_d;
            break;
         default:
            vassert(0);
         }
         accumulateFFLAGS(irsb,
                          mkIRExprCCall(Ity_I32, 0 /*regparms*/, helper_name,
                                        helper_addr,
                                        mkIRExprVec_2(mkexpr(a1), mkexpr(a2))));
         DIP("%s.d %s, %s, %s\n", name, nameIReg(rd), nameFReg(rs1),
             nameFReg(rs2));
         return True;
      }
   }

   /* ------------------ fclass.d rd, rs1 ------------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(14, 12) == 0b001 &&
       INSN(24, 20) == 0b00000 && INSN(31, 25) == 0b1110001) {
      UInt rd  = INSN(11, 7);
      UInt rs1 = INSN(19, 15);
      if (rd != 0)
         putIReg64(irsb, rd,
                   mkIRExprCCall(Ity_I64, 0 /*regparms*/,
                                 "riscv64g_calculate_fclass_d",
                                 riscv64g_calculate_fclass_d,
                                 mkIRExprVec_1(getFReg64(rs1))));
      DIP("fclass.d %s, %s\n", nameIReg(rd), nameFReg(rs1));
      return True;
   }

   /* -------------- fcvt.{w,wu}.d rd, rs1, rm -------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(24, 21) == 0b0000 &&
       INSN(31, 25) == 0b1100001) {
      UInt   rd        = INSN(11, 7);
      UInt   rm        = INSN(14, 12);
      UInt   rs1       = INSN(19, 15);
      Bool   is_signed = INSN(20, 20) == 0b0;
      IRTemp rm_RISCV, rm_IR;
      mk_get_rounding_mode(irsb, &rm_RISCV, &rm_IR, rm);
      IRTemp a1 = newTemp(irsb, Ity_F64);
      assign(irsb, a1, getFReg64(rs1));
      if (rd != 0)
         putIReg32(irsb, rd,
                   binop(is_signed ? Iop_F64toI32S : Iop_F64toI32U,
                         mkexpr(rm_IR), mkexpr(a1)));
      accumulateFFLAGS(
         irsb, mkIRExprCCall(Ity_I32, 0 /*regparms*/,
                             is_signed ? "riscv64g_calculate_fflags_fcvt_w_d"
                                       : "riscv64g_calculate_fflags_fcvt_wu_d",
                             is_signed ? riscv64g_calculate_fflags_fcvt_w_d
                                       : riscv64g_calculate_fflags_fcvt_wu_d,
                             mkIRExprVec_2(mkexpr(a1), mkexpr(rm_RISCV))));
      DIP("fcvt.w%s.d %s, %s%s\n", is_signed ? "" : "u", nameIReg(rd),
          nameFReg(rs1), nameRMOperand(rm));
      return True;
   }

   /* -------------- fcvt.d.{w,wu} rd, rs1, rm -------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(24, 21) == 0b0000 &&
       INSN(31, 25) == 0b1101001) {
      UInt rd  = INSN(11, 7);
      UInt rm  = INSN(14, 12); /* Ignored as the result is always exact. */
      UInt rs1 = INSN(19, 15);
      Bool is_signed = INSN(20, 20) == 0b0;
      putFReg64(
         irsb, rd,
         unop(is_signed ? Iop_I32StoF64 : Iop_I32UtoF64, getIReg32(rs1)));
      DIP("fcvt.d.w%s %s, %s%s\n", is_signed ? "" : "u", nameFReg(rd),
          nameIReg(rs1), nameRMOperand(rm));
      return True;
   }

   /* -------------- fcvt.{l,lu}.d rd, rs1, rm -------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(24, 21) == 0b0001 &&
       INSN(31, 25) == 0b1100001) {
      UInt   rd        = INSN(11, 7);
      UInt   rm        = INSN(14, 12);
      UInt   rs1       = INSN(19, 15);
      Bool   is_signed = INSN(20, 20) == 0b0;
      IRTemp rm_RISCV, rm_IR;
      mk_get_rounding_mode(irsb, &rm_RISCV, &rm_IR, rm);
      IRTemp a1 = newTemp(irsb, Ity_F64);
      assign(irsb, a1, getFReg64(rs1));
      if (rd != 0)
         putIReg64(irsb, rd,
                   binop(is_signed ? Iop_F64toI64S : Iop_F64toI64U,
                         mkexpr(rm_IR), mkexpr(a1)));
      accumulateFFLAGS(
         irsb, mkIRExprCCall(Ity_I32, 0 /*regparms*/,
                             is_signed ? "riscv64g_calculate_fflags_fcvt_l_d"
                                       : "riscv64g_calculate_fflags_fcvt_lu_d",
                             is_signed ? riscv64g_calculate_fflags_fcvt_l_d
                                       : riscv64g_calculate_fflags_fcvt_lu_d,
                             mkIRExprVec_2(mkexpr(a1), mkexpr(rm_RISCV))));
      DIP("fcvt.l%s.d %s, %s%s\n", is_signed ? "" : "u", nameIReg(rd),
          nameFReg(rs1), nameRMOperand(rm));
      return True;
   }

   /* ------------------- fmv.x.d rd, rs1 ------------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(14, 12) == 0b000 &&
       INSN(24, 20) == 0b00000 && INSN(31, 25) == 0b1110001) {
      UInt rd  = INSN(11, 7);
      UInt rs1 = INSN(19, 15);
      if (rd != 0)
         putIReg64(irsb, rd, unop(Iop_ReinterpF64asI64, getFReg64(rs1)));
      DIP("fmv.x.d %s, %s\n", nameIReg(rd), nameFReg(rs1));
      return True;
   }

   /* -------------- fcvt.d.{l,lu} rd, rs1, rm -------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(24, 21) == 0b0001 &&
       INSN(31, 25) == 0b1101001) {
      UInt   rd        = INSN(11, 7);
      UInt   rm        = INSN(14, 12);
      UInt   rs1       = INSN(19, 15);
      Bool   is_signed = INSN(20, 20) == 0b0;
      IRTemp rm_RISCV, rm_IR;
      mk_get_rounding_mode(irsb, &rm_RISCV, &rm_IR, rm);
      IRTemp a1 = newTemp(irsb, Ity_I64);
      assign(irsb, a1, getIReg64(rs1));
      putFReg64(irsb, rd,
                binop(is_signed ? Iop_I64StoF64 : Iop_I64UtoF64, mkexpr(rm_IR),
                      mkexpr(a1)));
      accumulateFFLAGS(
         irsb, mkIRExprCCall(Ity_I32, 0 /*regparms*/,
                             is_signed ? "riscv64g_calculate_fflags_fcvt_d_l"
                                       : "riscv64g_calculate_fflags_fcvt_d_lu",
                             is_signed ? riscv64g_calculate_fflags_fcvt_d_l
                                       : riscv64g_calculate_fflags_fcvt_d_lu,
                             mkIRExprVec_2(mkexpr(a1), mkexpr(rm_RISCV))));
      DIP("fcvt.d.l%s %s, %s%s\n", is_signed ? "" : "u", nameFReg(rd),
          nameIReg(rs1), nameRMOperand(rm));
      return True;
   }

   /* ------------------- fmv.d.x rd, rs1 ------------------- */
   if (INSN(6, 0) == 0b1010011 && INSN(14, 12) == 0b000 &&
       INSN(24, 20) == 0b00000 && INSN(31, 25) == 0b1111001) {
      UInt rd  = INSN(11, 7);
      UInt rs1 = INSN(19, 15);
      putFReg64(irsb, rd, unop(Iop_ReinterpI64asF64, getIReg64(rs1)));
      DIP("fmv.d.x %s, %s\n", nameFReg(rd), nameIReg(rs1));
      return True;
   }

   return False;
}

static Bool dis_RV64Zicsr(/*MB_OUT*/ DisResult* dres,
                          /*OUT*/ IRSB*         irsb,
                          UInt                  insn)
{
   /* ------------ RV64Zicsr standard extension ------------- */

   /* ----------------- csrrw rd, csr, rs1 ------------------ */
   if (INSN(6, 0) == 0b1110011 && INSN(14, 12) == 0b001) {
      UInt rd  = INSN(11, 7);
      UInt rs1 = INSN(19, 15);
      UInt csr = INSN(31, 20);
      if (csr != 0x001 && csr != 0x002 && csr != 0x003) {
         /* Invalid CSRRW, fall through. */
      } else {
         switch (csr) {
         case 0x001: {
            /* fflags */
            IRTemp fcsr = newTemp(irsb, Ity_I32);
            assign(irsb, fcsr, getFCSR());
            if (rd != 0)
               putIReg64(irsb, rd,
                         unop(Iop_32Uto64,
                              binop(Iop_And32, mkexpr(fcsr), mkU32(0x1f))));
            putFCSR(irsb,
                    binop(Iop_Or32,
                          binop(Iop_And32, mkexpr(fcsr), mkU32(0xffffffe0)),
                          binop(Iop_And32, getIReg32(rs1), mkU32(0x1f))));
            break;
         }
         case 0x002: {
            /* frm */
            IRTemp fcsr = newTemp(irsb, Ity_I32);
            assign(irsb, fcsr, getFCSR());
            if (rd != 0)
               putIReg64(
                  irsb, rd,
                  unop(Iop_32Uto64,
                       binop(Iop_And32, binop(Iop_Shr32, mkexpr(fcsr), mkU8(5)),
                             mkU32(0x7))));
            putFCSR(irsb,
                    binop(Iop_Or32,
                          binop(Iop_And32, mkexpr(fcsr), mkU32(0xffffff1f)),
                          binop(Iop_Shl32,
                                binop(Iop_And32, getIReg32(rs1), mkU32(0x7)),
                                mkU8(5))));
            break;
         }
         case 0x003: {
            /* fcsr */
            IRTemp fcsr = newTemp(irsb, Ity_I32);
            assign(irsb, fcsr, getFCSR());
            if (rd != 0)
               putIReg64(irsb, rd, unop(Iop_32Uto64, mkexpr(fcsr)));
            putFCSR(irsb, binop(Iop_And32, getIReg32(rs1), mkU32(0xff)));
            break;
         }
         default:
            vassert(0);
         }
         DIP("csrrs %s, %s, %s\n", nameIReg(rd), nameCSR(csr), nameIReg(rs1));
         return True;
      }
   }

   /* ----------------- csrrs rd, csr, rs1 ------------------ */
   if (INSN(6, 0) == 0b1110011 && INSN(14, 12) == 0b010) {
      UInt rd  = INSN(11, 7);
      UInt rs1 = INSN(19, 15);
      UInt csr = INSN(31, 20);
      if (csr != 0x001 && csr != 0x002 && csr != 0x003 && csr != 0xc20
            && csr != 0xc22) {
         /* Invalid CSRRS, fall through. */
      } else {
         switch (csr) {
         case 0x001: {
            /* fflags */
            IRTemp fcsr = newTemp(irsb, Ity_I32);
            assign(irsb, fcsr, getFCSR());
            if (rd != 0)
               putIReg64(irsb, rd,
                         unop(Iop_32Uto64,
                              binop(Iop_And32, mkexpr(fcsr), mkU32(0x1f))));
            putFCSR(irsb, binop(Iop_Or32, mkexpr(fcsr),
                                binop(Iop_And32, getIReg32(rs1), mkU32(0x1f))));
            break;
         }
         case 0x002: {
            /* frm */
            IRTemp fcsr = newTemp(irsb, Ity_I32);
            assign(irsb, fcsr, getFCSR());
            if (rd != 0)
               putIReg64(
                  irsb, rd,
                  unop(Iop_32Uto64,
                       binop(Iop_And32, binop(Iop_Shr32, mkexpr(fcsr), mkU8(5)),
                             mkU32(0x7))));
            putFCSR(irsb,
                    binop(Iop_Or32, mkexpr(fcsr),
                          binop(Iop_Shl32,
                                binop(Iop_And32, getIReg32(rs1), mkU32(0x7)),
                                mkU8(5))));
            break;
         }
         case 0x003: {
            /* fcsr */
            IRTemp fcsr = newTemp(irsb, Ity_I32);
            assign(irsb, fcsr, getFCSR());
            if (rd != 0)
               putIReg64(irsb, rd, unop(Iop_32Uto64, mkexpr(fcsr)));
            putFCSR(irsb, binop(Iop_Or32, mkexpr(fcsr),
                                binop(Iop_And32, getIReg32(rs1), mkU32(0xff))));
            break;
         }
         case 0xc20: {
            /* vl */
            IRTemp vl = newTemp(irsb, Ity_I64);
            assign(irsb, vl, IRExpr_Get(OFFB_VL, Ity_I64));
            if (rd != 0)
               putIReg64(irsb, rd, mkexpr(vl));
            vassert(rs1 == 0);
            break;
         }
         case 0xc22: {
            if (rd != 0)
               putIReg64(irsb, rd, mkU64(VLEN / 8));
            vassert(rs1 == 0);
            break;
         }
         default:
            vassert(0);
         }
         DIP("csrrs %s, %s, %s\n", nameIReg(rd), nameCSR(csr), nameIReg(rs1));
         return True;
      }
   }

   return False;
}

static inline Long sext_slice_ulong(ULong value, UInt bmax, UInt bmin)
{
   return ((Long)value) << (63 - bmax) >> (63 - (bmax - bmin));
}

#define MAX_VL  (-1ULL)
#define KEEP_VL (-2ULL)

static ULong helper_vsetvl(VexGuestRISCV64State* guest, ULong avl, ULong vtype)
{
   UInt sew = SLICE_UInt(vtype, 5, 3);
   Int lmul = sext_slice_ulong(vtype, 2, 0);

   ULong vlmax = VLEN >> (sew + 3 - lmul);
   ULong vl = guest->guest_vl;
   if (avl != KEEP_VL)
      vl = (avl < vlmax) ? avl : vlmax;

   guest->guest_vl = vl;
   guest->guest_vtype = vtype;
   guest->guest_vlmax = vlmax;

   invalidateFastCache();

   DIP("vsetvl - vl: %llu, sew: 0x%x, lmul: %d, avl: %llu, vtype: %llx\n",
       vl, sew, lmul, avl, vtype);

   return vl;
}

static Bool dis_vsetvl(/*MB_OUT*/ DisResult* dres,
                       /*OUT*/ IRSB*         irsb,
                       UInt                  insn,
                       Addr                  guest_pc_curr_instr)
{
   UInt rd = INSN(11, 7);
   IRExpr* avl;
   IRExpr* vtype;

   if (INSN(31, 30) == 0b11) {  // vsetivli
      UInt uimm = INSN(19, 15);
      Int zimm = INSN(29, 20);
      avl = mkU64(uimm);
      vtype = mkU64(zimm);
   } else if (INSN(31, 31) == 0b0 || INSN(31, 25) == 0b1000000) {
      UInt rs1 = INSN(19, 15);
      if (rs1 != 0) {
         avl = getIReg64(rs1);
      } else if (rd == 0) {
         avl = mkU64(KEEP_VL);
      } else {
         avl = mkU64(MAX_VL);
      }

      if (INSN(31, 31) == 0b0) {  // vsetvli
         Int zimm = INSN(30, 20);
         vtype = mkU64(zimm);
      } else {  // vsetvl
         UInt rs2 = INSN(24, 20);
         vtype = getIReg64(rs2);
      }
   } else {
      vassert(0);
   }

   IRTemp vl = newTemp(irsb, Ity_I64);
   IRDirty *d = unsafeIRDirty_1_N(vl,
         0,
         "helper_vsetvl",
         &helper_vsetvl,
         mkIRExprVec_3(IRExpr_GSPTR(), avl, vtype));

   d->nFxState = 2;
   vex_bzero(&d->fxState, sizeof(d->fxState));
   d->fxState[0].fx = Ifx_Write;
   d->fxState[0].offset = OFFB_VL;
   d->fxState[0].size = sizeof(ULong);
   d->fxState[1].fx = Ifx_Write;
   d->fxState[1].offset = OFFB_VTYPE;
   d->fxState[1].size = sizeof(ULong);

   stmt(irsb, IRStmt_Dirty(d));

   if (rd != 0) {
      putIReg64(irsb, rd, mkexpr(vl));
   }

   putPC(irsb, mkU64(guest_pc_curr_instr + 4));
   dres->whatNext    = Dis_StopHere;
   dres->jk_StopHere = Ijk_SyncupEnv;

   return True;
}

// return sew in bits
static UInt get_sew(VexGuestRISCV64State* guest)
{
   UInt raw_sew = SLICE_UInt(guest->guest_vtype, 5, 3);
   switch (raw_sew) {
   case 0b000: return 8;
   case 0b001: return 16;
   case 0b010: return 32;
   case 0b011: return 64;
   default: vassert(0);
   }
}

static Int sewToIndex(UInt sew)
{
   Int index = -1;
   switch (sew) {
      case  8: index = 0;  break;
      case 16: index = 1; break;
      case 32: index = 2; break;
      case 64: index = 3 ; break;
      default: vassert(0);
   }
   return index;
}

typedef enum {
   RVV_V = 0,
   RVV_X,
   RVV_I,
   RVV_W,
} rvvRegType;

static Bool dis_rvv_iext(/*MB_OUT*/ DisResult* dres,
                         /*OUT*/ IRSB*         irsb,
                         UInt                  insn,
                         Addr                  guest_pc_curr_instr,
                         VexGuestRISCV64State* guest)
{
   UInt vm = INSN(25, 25);
   UInt vs2 = INSN(24, 20);
   UInt s1 = INSN(19, 15);
   UInt vd = INSN(11, 7);

   UInt vma = SLICE_UInt(guest->guest_vtype, 7, 7);
   UInt sew = get_sew(guest);
   Int index = sewToIndex(sew);
   UInt vl = guest->guest_vl;
   IRType dst_ty = typeofVecIR(vl, Ity_VLen8 + index);
   IRType src_ty;

   vassert(sew >= 16);

   IROp op;
   switch (s1) {
   case 0b00110: op = Iop_VZext_vf2_16 + index - 1; src_ty = dst_ty - 1; break;
   case 0b00100: op = Iop_VZext_vf4_32 + index - 2; src_ty = dst_ty - 2; break;
   case 0b00010: op = Iop_VZext_vf8_64 + index - 3; src_ty = dst_ty - 3; break;
   case 0b00111: op = Iop_VSext_vf2_16 + index - 1; src_ty = dst_ty - 1; break;
   case 0b00101: op = Iop_VSext_vf4_32 + index - 2; src_ty = dst_ty - 2; break;
   case 0b00011: op = Iop_VSext_vf8_64 + index - 3; src_ty = dst_ty - 3; break;
   default: vassert(0);
   }

   IRExpr* res = unop(opofVecIR(vl, op), getVReg(vs2, 0, src_ty));
   if (vm == 0) {
      IRType mask_ty = typeofVecIR(vl, Ity_VLen1);
      IRExpr* mask = unop(opofVecIR(vl, Iop_VExpandBitsTo_8 + index),
                          getVReg(0 /*v0*/, 0, mask_ty));

      if (vma == 0) { // undisturbed, read it first
         IRExpr* origin = getVReg(vd, 0, dst_ty);
         IRExpr* inactive = binop(opofVecIR(vl, Iop_VAnd_vv_8 + index),
                                  unop(opofVecIR(vl, Iop_VNot_8 + index), mask),
                                  origin);
         IRExpr* active = binop(opofVecIR(vl, Iop_VAnd_vv_8 + index), mask, res);
         res = binop(opofVecIR(vl, Iop_VOr_vv_8 + index), active, inactive);
      } else { // agnostic, set to 1
         res = binop(opofVecIR(vl, Iop_VOr_vv_8 + index),
                     unop(opofVecIR(vl, Iop_VNot_8 + index), mask),
                     res);
      }
   }
   putVReg(irsb, vd, 0, res);

   return True;
}

/*
 * Two source version of indepedent elements ops such as vadd, where every
 * element is calculated indepedently.
 *    [dst], src2, src1
 *    v,     v/w,  v/x/i
 */
static Bool dis_rvv2_vw_vxi(/*MB_OUT*/ DisResult* dres,
                            /*OUT*/ IRSB*         irsb,
                            UInt                  insn,
                            Addr                  guest_pc_curr_instr,
                            VexGuestRISCV64State* guest,
                            IROp                  base_op,
                            rvvRegType            s2_type,
                            rvvRegType            s1_type)
{
   UInt vm = INSN(25, 25);
   UInt vs2 = INSN(24, 20);
   UInt s1 = INSN(19, 15);
   UInt vd = INSN(11, 7);

   UInt vma = SLICE_UInt(guest->guest_vtype, 7, 7);
   UInt sew = get_sew(guest);
   Int index = sewToIndex(sew);
   UInt vl = guest->guest_vl;
   IRType ty = typeofVecIR(vl, Ity_VLen8 + index);

   IRExpr* es1 = NULL;
   if (s1_type == RVV_V) {
      es1 = getVReg(s1, 0, ty);
   } else if (s1_type == RVV_X) {
      es1 = getIReg64(s1);
   } else if (s1_type == RVV_I) {
      Long imm = sext_slice_ulong(insn, 19, 15);
      es1 = mkU64(imm);
   } else {
      vassert(0);
   }

   IRExpr* es2 = NULL;
   if (s2_type == RVV_V) {
      es2 = getVReg(vs2, 0, ty);
   } else if (s2_type == RVV_W) {
      vassert(sew != 64);
      Int index2 = sewToIndex(sew * 2);
      IRType type2 = typeofVecIR(vl, Ity_VLen8 + index2);
      es2 = getVReg(vs2, 0, type2);
   } else {
      vassert(0);
   }

   IRExpr* res = binop(opofVecIR(vl, base_op + index), es1, es2);
   if (vm == 0) {
      IRType mask_ty = typeofVecIR(vl, Ity_VLen1);
      IRExpr* mask = unop(opofVecIR(vl, Iop_VExpandBitsTo_8 + index),
                          getVReg(0 /*v0*/, 0, mask_ty));

      if (vma == 0) { // undisturbed, read it first
         IRExpr* origin = getVReg(vd, 0, ty);
         IRExpr* inactive = binop(opofVecIR(vl, Iop_VAnd_vv_8 + index),
                                  unop(opofVecIR(vl, Iop_VNot_8 + index), mask),
                                  origin);
         IRExpr* active = binop(opofVecIR(vl, Iop_VAnd_vv_8 + index), mask, res);
         res = binop(opofVecIR(vl, Iop_VOr_vv_8 + index), active, inactive);
      } else { // agnostic, set to 1
         res = binop(opofVecIR(vl, Iop_VOr_vv_8 + index),
                     unop(opofVecIR(vl, Iop_VNot_8 + index), mask),
                     res);
      }
   }
   putVReg(irsb, vd, 0, res);

   return True;
}

static inline
Bool dis_rvv2_v_v(/*MB_OUT*/ DisResult* dres,
                  /*OUT*/ IRSB*         irsb,
                  UInt                  insn,
                  Addr                  guest_pc_curr_instr,
                  VexGuestRISCV64State* guest,
                  IROp                  base_op)
{
   return dis_rvv2_vw_vxi(dres, irsb, insn, guest_pc_curr_instr, guest, base_op, RVV_V, RVV_V);
}

static inline
Bool dis_rvv2_v_x(/*MB_OUT*/ DisResult* dres,
                  /*OUT*/ IRSB*         irsb,
                  UInt                  insn,
                  Addr                  guest_pc_curr_instr,
                  VexGuestRISCV64State* guest,
                  IROp                  base_op)
{
   return dis_rvv2_vw_vxi(dres, irsb, insn, guest_pc_curr_instr, guest, base_op, RVV_V, RVV_X);
}

static inline
Bool dis_rvv2_v_i(/*MB_OUT*/ DisResult* dres,
                  /*OUT*/ IRSB*         irsb,
                  UInt                  insn,
                  Addr                  guest_pc_curr_instr,
                  VexGuestRISCV64State* guest,
                  IROp                  base_op)
{
   return dis_rvv2_vw_vxi(dres, irsb, insn, guest_pc_curr_instr, guest, base_op, RVV_V, RVV_I);
}

static inline
Bool dis_rvv2_w_v(/*MB_OUT*/ DisResult* dres,
                  /*OUT*/ IRSB*         irsb,
                  UInt                  insn,
                  Addr                  guest_pc_curr_instr,
                  VexGuestRISCV64State* guest,
                  IROp                  base_op)
{
   return dis_rvv2_vw_vxi(dres, irsb, insn, guest_pc_curr_instr, guest, base_op, RVV_W, RVV_V);
}

static inline
Bool dis_rvv2_w_x(/*MB_OUT*/ DisResult* dres,
                  /*OUT*/ IRSB*         irsb,
                  UInt                  insn,
                  Addr                  guest_pc_curr_instr,
                  VexGuestRISCV64State* guest,
                  IROp                  base_op)
{
   return dis_rvv2_vw_vxi(dres, irsb, insn, guest_pc_curr_instr, guest, base_op, RVV_W, RVV_X);
}

static inline
Bool dis_rvv2_w_i(/*MB_OUT*/ DisResult* dres,
                  /*OUT*/ IRSB*         irsb,
                  UInt                  insn,
                  Addr                  guest_pc_curr_instr,
                  VexGuestRISCV64State* guest,
                  IROp                  base_op)
{
   return dis_rvv2_vw_vxi(dres, irsb, insn, guest_pc_curr_instr, guest, base_op, RVV_W, RVV_I);
}

/*
 * Two source version of indepedent elements ops such as wvadd, where every
 * element is calculated indepedently.
 *    [dst], src2, src1
 *    w,     v/w,  v/x
 */
static Bool dis_rvv2w_vw_vx(/*MB_OUT*/ DisResult* dres,
                             /*OUT*/ IRSB*         irsb,
                             UInt                  insn,
                             Addr                  guest_pc_curr_instr,
                             VexGuestRISCV64State* guest,
                             IROp                  base_op,
                             rvvRegType            s2_type,
                             rvvRegType            s1_type)
{
   UInt vm = INSN(25, 25);
   UInt vs2 = INSN(24, 20);
   UInt s1 = INSN(19, 15);
   UInt vd = INSN(11, 7);

   UInt vma = SLICE_UInt(guest->guest_vtype, 7, 7);
   UInt sew = get_sew(guest);
   Int index = sewToIndex(sew);
   UInt vl = guest->guest_vl;
   Int dst_index = index + 1;
   IRType ty = typeofVecIR(vl, Ity_VLen8 + index);
   IRType dst_ty = ty + 1;

   IRExpr* es2 = NULL;
   if (s2_type == RVV_V) {
      es2 = getVReg(vs2, 0, ty);
   } else if (s2_type == RVV_W) {
      getVReg(vs2, 0, ty + 1);
   } else {
      vassert(0);
   }

   IRExpr* es1 = NULL;
   if (s1_type == RVV_V) {
      es1 = getVReg(s1, 0, ty);
   } else if (s1_type == RVV_X) {
      es1 = getIReg64(s1);
   } else {
      vassert(0);
   }

   IRExpr* res = binop(opofVecIR(vl, base_op + index), es1, es2);
   if (vm == 0) {
      IRType mask_ty = typeofVecIR(vl, Ity_VLen1);
      IRExpr* mask = unop(opofVecIR(vl, Iop_VExpandBitsTo_8 + dst_index),
                          getVReg(0 /*v0*/, 0, mask_ty));

      if (vma == 0) { // undisturbed, read it first
         IRExpr* origin = getVReg(vd, 0, dst_ty);
         IRExpr* inactive = binop(opofVecIR(vl, Iop_VAnd_vv_8 + dst_index),
                                  unop(opofVecIR(vl, Iop_VNot_8 + dst_index), mask),
                                  origin);
         IRExpr* active = binop(opofVecIR(vl, Iop_VAnd_vv_8 + dst_index), mask, res);
         res = binop(opofVecIR(vl, Iop_VOr_vv_8 + dst_index), active, inactive);
      } else { // agnostic, set to 1
         res = binop(opofVecIR(vl, Iop_VOr_vv_8 + dst_index),
                     unop(opofVecIR(vl, Iop_VNot_8 + dst_index), mask),
                     res);
      }
   }
   putVReg(irsb, vd, 0, res);

   return True;
}

static inline
Bool dis_rvv2w_v_v(/*MB_OUT*/ DisResult* dres,
                    /*OUT*/ IRSB*         irsb,
                    UInt                  insn,
                    Addr                  guest_pc_curr_instr,
                    VexGuestRISCV64State* guest,
                    IROp                  base_op)
{
   return dis_rvv2w_vw_vx(dres, irsb, insn, guest_pc_curr_instr, guest, base_op, RVV_V, RVV_V);
}

static inline
Bool dis_rvv2w_v_x(/*MB_OUT*/ DisResult* dres,
                    /*OUT*/ IRSB*         irsb,
                    UInt                  insn,
                    Addr                  guest_pc_curr_instr,
                    VexGuestRISCV64State* guest,
                    IROp                  base_op)
{
   return dis_rvv2w_vw_vx(dres, irsb, insn, guest_pc_curr_instr, guest, base_op, RVV_V, RVV_X);
}

static inline
Bool dis_rvv2w_w_v(/*MB_OUT*/ DisResult* dres,
                    /*OUT*/ IRSB*         irsb,
                    UInt                  insn,
                    Addr                  guest_pc_curr_instr,
                    VexGuestRISCV64State* guest,
                    IROp                  base_op)
{
   return dis_rvv2w_vw_vx(dres, irsb, insn, guest_pc_curr_instr, guest, base_op, RVV_W, RVV_V);
}

static inline
Bool dis_rvv2w_w_x(/*MB_OUT*/ DisResult* dres,
                    /*OUT*/ IRSB*         irsb,
                    UInt                  insn,
                    Addr                  guest_pc_curr_instr,
                    VexGuestRISCV64State* guest,
                    IROp                  base_op)
{
   return dis_rvv2w_vw_vx(dres, irsb, insn, guest_pc_curr_instr, guest, base_op, RVV_W, RVV_X);
}

/*
 * Three source version of indepedent elements ops such as vmadd, where every
 * element is calculated indepedently.
 *    [dst], src2, src1, src3(dst)
 *    v,     v,    v/x,  v
 */
static Bool dis_rvv3_v_vx(/*MB_OUT*/ DisResult* dres,
                          /*OUT*/ IRSB*         irsb,
                          UInt                  insn,
                          Addr                  guest_pc_curr_instr,
                          VexGuestRISCV64State* guest,
                          IROp                  base_op,
                          rvvRegType            s1_type)
{
   UInt vm = INSN(25, 25);
   UInt vs2 = INSN(24, 20);
   UInt s1 = INSN(19, 15);
   UInt vd = INSN(11, 7);

   UInt vma = SLICE_UInt(guest->guest_vtype, 7, 7);
   UInt sew = get_sew(guest);
   Int index = sewToIndex(sew);
   UInt vl = guest->guest_vl;
   IRType ty = typeofVecIR(vl, Ity_VLen8 + index);

   IRExpr* es1 = NULL;
   if (s1_type == RVV_V) {
      es1 = getVReg(s1, 0, ty);
   } else if (s1_type == RVV_X) {
      es1 = getIReg64(s1);
   } else if (s1_type == RVV_I) {
      vassert(0);
   }

   IRExpr* res = triop(opofVecIR(vl, base_op + index),
                       es1, getVReg(vs2, 0, ty), getVReg(vd, 0, ty));
   if (vm == 0) {
      IRType mask_ty = typeofVecIR(vl, Ity_VLen1);
      IRExpr* mask = unop(opofVecIR(vl, Iop_VExpandBitsTo_8 + index),
                          getVReg(0 /*v0*/, 0, mask_ty));

      if (vma == 0) { // undisturbed, read it first
         IRExpr* origin = getVReg(vd, 0, ty);
         IRExpr* inactive = binop(opofVecIR(vl, Iop_VAnd_vv_8 + index),
                                  unop(opofVecIR(vl, Iop_VNot_8 + index), mask),
                                  origin);
         IRExpr* active = binop(opofVecIR(vl, Iop_VAnd_vv_8 + index), mask, res);
         res = binop(opofVecIR(vl, Iop_VOr_vv_8 + index), active, inactive);
      } else { // agnostic, set to 1
         res = binop(opofVecIR(vl, Iop_VOr_vv_8 + index),
                     unop(opofVecIR(vl, Iop_VNot_8 + index), mask),
                     res);
      }
   }
   putVReg(irsb, vd, 0, res);

   return True;
}

static inline
Bool dis_rvv3_v_v(/*MB_OUT*/ DisResult* dres,
                  /*OUT*/ IRSB*         irsb,
                  UInt                  insn,
                  Addr                  guest_pc_curr_instr,
                  VexGuestRISCV64State* guest,
                  IROp                  base_op)
{
   return dis_rvv3_v_vx(dres, irsb, insn, guest_pc_curr_instr, guest, base_op, RVV_V);
}

static inline
Bool dis_rvv3_v_x(/*MB_OUT*/ DisResult* dres,
                  /*OUT*/ IRSB*         irsb,
                  UInt                  insn,
                  Addr                  guest_pc_curr_instr,
                  VexGuestRISCV64State* guest,
                  IROp                  base_op)
{
   return dis_rvv3_v_vx(dres, irsb, insn, guest_pc_curr_instr, guest, base_op, RVV_X);
}

/*
 * Three source version of indepedent elements ops such as wmadd, where every
 * element is calculated indepedently.
 *    [dst], src2, src1, src3(dst)
 *    w,     v,    v/x,  w
 */
static Bool dis_rvv3w_v_vx(/*MB_OUT*/ DisResult* dres,
                            /*OUT*/ IRSB*         irsb,
                            UInt                  insn,
                            Addr                  guest_pc_curr_instr,
                            VexGuestRISCV64State* guest,
                            IROp                  base_op,
                            rvvRegType            s1_type)
{
   UInt vm = INSN(25, 25);
   UInt vs2 = INSN(24, 20);
   UInt s1 = INSN(19, 15);
   UInt vd = INSN(11, 7);

   UInt vma = SLICE_UInt(guest->guest_vtype, 7, 7);
   UInt sew = get_sew(guest);
   Int index = sewToIndex(sew);
   UInt vl = guest->guest_vl;
   Int dst_index = index + 1;
   IRType ty = typeofVecIR(vl, Ity_VLen8 + index);
   IRType dst_ty = ty + 1;

   IRExpr* es1 = NULL;
   if (s1_type == RVV_V) {
      es1 = getVReg(s1, 0, ty);
   } else if (s1_type == RVV_X) {
      es1 = getIReg64(s1);
   } else {
      vassert(0);
   }

   IRExpr* res = triop(opofVecIR(vl, base_op + index),
                       es1, getVReg(vs2, 0, ty), getVReg(vd, 0, dst_ty));
   if (vm == 0) {
      IRType mask_ty = typeofVecIR(vl, Ity_VLen1);
      IRExpr* mask = unop(opofVecIR(vl, Iop_VExpandBitsTo_8 + dst_index),
                          getVReg(0 /*v0*/, 0, mask_ty));

      if (vma == 0) { // undisturbed, read it first
         IRExpr* origin = getVReg(vd, 0, dst_ty);
         IRExpr* inactive = binop(opofVecIR(vl, Iop_VAnd_vv_8 + dst_index),
                                  unop(opofVecIR(vl, Iop_VNot_8 + dst_index), mask),
                                  origin);
         IRExpr* active = binop(opofVecIR(vl, Iop_VAnd_vv_8 + dst_index), mask, res);
         res = binop(opofVecIR(vl, Iop_VOr_vv_8 + dst_index), active, inactive);
      } else { // agnostic, set to 1
         res = binop(opofVecIR(vl, Iop_VOr_vv_8 + dst_index),
                     unop(opofVecIR(vl, Iop_VNot_8 + dst_index), mask),
                     res);
      }
   }
   putVReg(irsb, vd, 0, res);

   return True;
}

static inline
Bool dis_rvv3w_v_v(/*MB_OUT*/ DisResult* dres,
                    /*OUT*/ IRSB*         irsb,
                    UInt                  insn,
                    Addr                  guest_pc_curr_instr,
                    VexGuestRISCV64State* guest,
                    IROp                  base_op)
{
   return dis_rvv3w_v_vx(dres, irsb, insn, guest_pc_curr_instr, guest, base_op, RVV_V);
}

static inline
Bool dis_rvv3w_v_x(/*MB_OUT*/ DisResult* dres,
                    /*OUT*/ IRSB*         irsb,
                    UInt                  insn,
                    Addr                  guest_pc_curr_instr,
                    VexGuestRISCV64State* guest,
                    IROp                  base_op)
{
   return dis_rvv3w_v_vx(dres, irsb, insn, guest_pc_curr_instr, guest, base_op, RVV_X);
}

static Bool dis_rvv_vmvr(/*MB_OUT*/ DisResult* dres,
                         /*OUT*/ IRSB*         irsb,
                         UInt                  insn,
                         Addr                  guest_pc_curr_instr,
                         VexGuestRISCV64State* guest)
{
   UInt vm = INSN(25, 25);
   UInt vs = INSN(24, 20);
   UInt imm = INSN(19, 15);
   UInt vd = INSN(11, 7);
   UInt nreg = imm + 1;

   vassert(vm == 1);

   UInt sew = get_sew(guest);
   Int index = sewToIndex(sew);
   UInt vl = VLEN / sew;  // one vreg
   IRType ty = typeofVecIR(vl, Ity_VLen8 + index);

   for (int i = 0; i < nreg; ++i) {
      putVReg(irsb, vd++, 0, getVReg(vs++, 0, ty));
   }

   return True;
}

static Bool dis_rvv_vmv_x_s(/*MB_OUT*/ DisResult* dres,
                            /*OUT*/ IRSB*         irsb,
                            UInt                  insn,
                            Addr                  guest_pc_curr_instr,
                            VexGuestRISCV64State* guest)
{
   UInt vm = INSN(25, 25);
   UInt vs2 = INSN(24, 20);
   UInt rd = INSN(11, 7);

   vassert(vm == 1);

   UInt sew = get_sew(guest);
   Int index = sewToIndex(sew);
   Int ty = Ity_I8 + index;

   putIReg64(irsb, rd, widenSto64(ty, getVReg(vs2, 0, ty)));
   return True;
}

static Bool dis_rvv_vmv_s_x(/*MB_OUT*/ DisResult* dres,
                            /*OUT*/ IRSB*         irsb,
                            UInt                  insn,
                            Addr                  guest_pc_curr_instr,
                            VexGuestRISCV64State* guest)
{
   UInt vm = INSN(25, 25);
   UInt rs1 = INSN(19, 15);
   UInt vd = INSN(11, 7);

   vassert(vm == 1);

   UInt sew = get_sew(guest);
   Int index = sewToIndex(sew);
   Int ty = Ity_I8 + index;

   putVReg(irsb, vd, 0, narrowFrom64(ty, getIReg64(rs1)));
   return True;
}

static Bool dis_rvv_vmv_v_vxi(/*MB_OUT*/ DisResult* dres,
                              /*OUT*/ IRSB*         irsb,
                              UInt                  insn,
                              Addr                  guest_pc_curr_instr,
                              VexGuestRISCV64State* guest,
                              rvvRegType            s1_type)
{
   UInt vs2 = INSN(24, 20);
   UInt s1 = INSN(19, 15);
   UInt vd = INSN(11, 7);

   vassert(vs2 == 0);

   UInt sew = get_sew(guest);
   Int index = sewToIndex(sew);
   UInt vl = guest->guest_vl;
   IRType ty = typeofVecIR(vl, Ity_VLen8 + index);

   IRExpr* es1;
   IROp base_op;
   if (s1_type == RVV_V) {
      es1 = getVReg(s1, 0, ty);
      base_op = Iop_VMv_v_v_8;
   } else if (s1_type == RVV_X) {
      es1 = getIReg64(s1);
      base_op = Iop_VMv_v_x_8;
   } else if (s1_type == RVV_I) {
      es1 = mkU64(sext_slice_ulong(insn, 19, 15));
      base_op = Iop_VMv_v_i_8;
   }

   IRExpr* res = unop(opofVecIR(vl, base_op + index), es1);
   putVReg(irsb, vd, 0, res);

   return True;
}

static Bool dis_rvv_vmerge_vxi(/*MB_OUT*/ DisResult* dres,
                               /*OUT*/ IRSB*         irsb,
                               UInt                  insn,
                               Addr                  guest_pc_curr_instr,
                               VexGuestRISCV64State* guest,
                               rvvRegType            s1_type)
{
   UInt vs2 = INSN(24, 20);
   UInt s1 = INSN(19, 15);
   UInt vd = INSN(11, 7);

   UInt sew = get_sew(guest);
   Int index = sewToIndex(sew);
   UInt vl = guest->guest_vl;
   IRType ty = typeofVecIR(vl, Ity_VLen8 + index);

   IRExpr* es1;
   IROp base_op;
   if (s1_type == RVV_V) {
      es1 = getVReg(s1, 0, ty);
      base_op = Iop_VMerge_vvm_8;
   } else if (s1_type == RVV_X) {
      es1 = getIReg64(s1);
      base_op = Iop_VMerge_vxm_8;
   } else if (s1_type == RVV_I) {
      es1 = mkU64(sext_slice_ulong(insn, 19, 15));
      base_op = Iop_VMerge_vim_8;
   }

   IRExpr* mask = getVReg(0 /*v0*/, 0, typeofVecIR(vl, Ity_VLen1));
   IRExpr* res = triop(opofVecIR(vl, base_op + index),
                       es1, getVReg(vs2, 0, ty), mask);
   putVReg(irsb, vd, 0, res);

   return True;
}

static Bool dis_rvv_reduce(/*MB_OUT*/ DisResult* dres,
                           /*OUT*/ IRSB*         irsb,
                           UInt                  insn,
                           Addr                  guest_pc_curr_instr,
                           VexGuestRISCV64State* guest)
{
   UInt funct6 = INSN(31, 26);
   UInt vm = INSN(25, 25);
   UInt vs2 = INSN(24, 20);
   UInt vs1 = INSN(19, 15);
   UInt vd = INSN(11, 7);

   UInt sew = get_sew(guest);
   Int index = sewToIndex(sew);
   UInt vl = guest->guest_vl;
   IRType ty = typeofVecIR(vl, Ity_VLen8 + index);

   IROp base_op;
   switch (funct6) {
   case 0b000000: base_op = (vm == 0) ? Iop_VRedsum_vsm_8 : Iop_VRedsum_vs_8; break;
   case 0b000001: base_op = (vm == 0) ? Iop_VRedand_vsm_8 : Iop_VRedand_vs_8; break;
   case 0b000010: base_op = (vm == 0) ? Iop_VRedor_vsm_8 : Iop_VRedor_vs_8; break;
   case 0b000011: base_op = (vm == 0) ? Iop_VRedxor_vsm_8 : Iop_VRedxor_vs_8; break;
   case 0b000100: base_op = (vm == 0) ? Iop_VRedminu_vsm_8 : Iop_VRedminu_vs_8; break;
   case 0b000101: base_op = (vm == 0) ? Iop_VRedmin_vsm_8 : Iop_VRedmin_vs_8; break;
   case 0b000110: base_op = (vm == 0) ? Iop_VRedmaxu_vsm_8 : Iop_VRedmaxu_vs_8; break;
   case 0b000111: base_op = (vm == 0) ? Iop_VRedmax_vsm_8 : Iop_VRedmax_vs_8; break;
   /* NOTE: funct3 of vred and vwred is not same */
   case 0b110000: base_op = (vm == 0) ? Iop_VWredsumu_vsm_8 : Iop_VWredsumu_vs_8; break;
   case 0b110001: base_op = (vm == 0) ? Iop_VWredsum_vsm_8 : Iop_VWredsum_vs_8; break;
   default: vassert(0);
   }

   IRExpr* s1 = getVReg(vs1, 0, Ity_I64);
   IRExpr* s2 = getVReg(vs2, 0, ty);

   IRExpr* res;
   if (vm == 0) {
      IRType mask_ty = typeofVecIR(vl, Ity_VLen1);
      IRExpr* mask = getVReg(0 /*v0*/, 0, mask_ty);
      res = triop(opofVecIR(vl, base_op + index), s1, s2, mask);
   } else {
      res = binop(opofVecIR(vl, base_op + index), s1, s2);
   }

   putVReg(irsb, vd, 0, res);

   return True;
}

static Bool dis_rvv_m_m(/*MB_OUT*/ DisResult* dres,
                        /*OUT*/ IRSB*         irsb,
                        UInt                  insn,
                        Addr                  guest_pc_curr_instr,
                        VexGuestRISCV64State* guest,
                        IROp                  base_op)
{
   UInt vm = INSN(25, 25);
   UInt vs2 = INSN(24, 20);
   UInt vs1 = INSN(19, 15);
   UInt vd = INSN(11, 7);

   vassert(vm == 1);

   UInt vl = VLEN;
   IRType ty = typeofVecIR(vl, Ity_VLen1);

   IRExpr* s1 = getVReg(vs1, 0, ty);
   IRExpr* s2 = getVReg(vs2, 0, ty);

   putVReg(irsb, vd, 0, binop(opofVecIR(vl, base_op), s1, s2));

   return True;
}

static Bool dis_rvv_addsub_carry(/*MB_OUT*/ DisResult* dres,
                                 /*OUT*/ IRSB*         irsb,
                                 UInt                  insn,
                                 Addr                  guest_pc_curr_instr,
                                 VexGuestRISCV64State* guest)
{
   UInt funct6 = INSN(31, 26);
   UInt vm = INSN(25, 25);
   UInt vs2 = INSN(24, 20);
   UInt s1 = INSN(19, 15);
   UInt funct3 = INSN(14, 12);
   UInt vd = INSN(11, 7);

   UInt sew = get_sew(guest);
   Int index = sewToIndex(sew);
   UInt vl = guest->guest_vl;
   IRType ty = typeofVecIR(vl, Ity_VLen8 + index);
   rvvRegType s1_type;

   IROp base_op;
   switch (funct6) {
   case 0b010000: {
      switch (funct3) {
      case 0b000: base_op = Iop_VAdc_vvm_8; s1_type = RVV_V; break;
      case 0b100: base_op = Iop_VAdc_vxm_8; s1_type = RVV_X; break;
      case 0b011: base_op = Iop_VAdc_vim_8; s1_type = RVV_I; break;
      default: vassert(0);
      }
      break;
   }
   case 0b010001: {
      switch (funct3) {
      case 0b000: base_op = (vm == 0) ? Iop_VMadc_vvm_8 : Iop_VMadc_vv_8; s1_type = RVV_V; break;
      case 0b100: base_op = (vm == 0) ? Iop_VMadc_vxm_8 : Iop_VMadc_vx_8; s1_type = RVV_X; break;
      case 0b011: base_op = (vm == 0) ? Iop_VMadc_vim_8 : Iop_VMadc_vi_8; s1_type = RVV_I; break;
      default: vassert(0);
      }
      break;
   }
   case 0b010010: {
      switch (funct3) {
      case 0b000: base_op = Iop_VSbc_vvm_8; s1_type = RVV_V; break;
      case 0b100: base_op = Iop_VSbc_vxm_8; s1_type = RVV_X; break;
      default: vassert(0);
      }
      break;
   }
   case 0b010011: {
      switch (funct3) {
      case 0b000: base_op = (vm == 0) ? Iop_VMsbc_vvm_8 : Iop_VMsbc_vv_8; s1_type = RVV_V; break;
      case 0b100: base_op = (vm == 0) ? Iop_VMsbc_vxm_8 : Iop_VMsbc_vx_8; s1_type = RVV_X; break;
      default: vassert(0);
      }
      break;
   }
   default: vassert(0);
   }

   IRExpr* es1;
   if (s1_type == RVV_V) {
      es1 = getVReg(s1, 0, ty);
   } else if (s1_type == RVV_X) {
      es1 = getIReg64(s1);
   } else if (s1_type == RVV_I) {
      Long imm = sext_slice_ulong(insn, 19, 15);
      es1 = mkU64(imm);
   }

   IRExpr* es2 = getVReg(vs2, 0, ty);

   IRExpr* res;
   if (vm == 0) {
      IRType mask_ty = typeofVecIR(vl, Ity_VLen1);
      IRExpr* mask = getVReg(0 /*v0*/, 0, mask_ty);
      res = triop(opofVecIR(vl, base_op + index), es1, es2, mask);
   } else {
      res = binop(opofVecIR(vl, base_op + index), es1, es2);
   }
   putVReg(irsb, vd, 0, res);

   return True;
}

static Bool dis_vcpop_m(/*MB_OUT*/ DisResult* dres,
                        /*OUT*/ IRSB*         irsb,
                        UInt                  insn,
                        Addr                  guest_pc_curr_instr,
                        VexGuestRISCV64State* guest)
{
   UInt vm = INSN(25, 25);
   UInt vs2 = INSN(24, 20);
   UInt rd = INSN(11, 7);

   UInt vl = guest->guest_vl;
   IRType ty = typeofVecIR(vl, Ity_VLen1);

   IRExpr* e = getVReg(vs2, 0, ty);
   if (vm == 0) {
      IRExpr* mask = getVReg(0 /*v0*/, 0, ty);
      e = binop(opofVecIR(vl, Iop_VMand_mm), e, mask);
   }
   putIReg64(irsb, rd, unop(opofVecIR(vl, Iop_VCpop_m), e));

   return True;
}

static Bool dis_vfirst_m(/*MB_OUT*/ DisResult* dres,
                         /*OUT*/ IRSB*         irsb,
                         UInt                  insn,
                         Addr                  guest_pc_curr_instr,
                         VexGuestRISCV64State* guest)
{
   UInt vm = INSN(25, 25);
   UInt vs2 = INSN(24, 20);
   UInt rd = INSN(11, 7);

   UInt vl = guest->guest_vl;
   IRType ty = typeofVecIR(vl, Ity_VLen1);

   IRExpr* e = getVReg(vs2, 0, ty);
   if (vm == 0) {
      IRExpr* mask = getVReg(0 /*v0*/, 0, ty);
      e = binop(opofVecIR(vl, Iop_VMand_mm), e, mask);
   }
   putIReg64(irsb, rd, unop(opofVecIR(vl, Iop_VFirst_m), e));

   return True;
}

static Bool dis_vmsBIOf_m(/*MB_OUT*/ DisResult* dres,
                          /*OUT*/ IRSB*         irsb,
                          UInt                  insn,
                          Addr                  guest_pc_curr_instr,
                          VexGuestRISCV64State* guest,
                          IROp                  op)
{
   UInt vm = INSN(25, 25);
   UInt vs2 = INSN(24, 20);
   UInt vd = INSN(11, 7);

   UInt vl = guest->guest_vl;
   IRType ty = typeofVecIR(vl, Ity_VLen1);

   IRExpr* e = getVReg(vs2, 0, ty);
   if (vm == 0) {
      IRExpr* mask = getVReg(0 /*v0*/, 0, ty);
      e = binop(opofVecIR(vl, Iop_VMand_mm), e, mask);
   }
   putVReg(irsb, vd, 0, unop(opofVecIR(vl, op), e));

   return True;
}

static Bool dis_vid_v(/*MB_OUT*/ DisResult* dres,
                      /*OUT*/ IRSB*         irsb,
                      UInt                  insn,
                      Addr                  guest_pc_curr_instr,
                      VexGuestRISCV64State* guest)
{
   UInt vm = INSN(25, 25);
   UInt vd = INSN(11, 7);

   UInt vma = SLICE_UInt(guest->guest_vtype, 7, 7);
   UInt sew = get_sew(guest);
   Int index = sewToIndex(sew);
   UInt vl = guest->guest_vl;
   IRType ty = typeofVecIR(vl, Ity_VLen8 + index);

   IRExpr* res = unop(opofVecIR(vl, Iop_VId_v_8 + index), mkU64(0));
   if (vm == 0) {
      IRType mask_ty = typeofVecIR(vl, Ity_VLen1);
      IRExpr* mask = unop(opofVecIR(vl, Iop_VExpandBitsTo_8 + index),
                          getVReg(0 /*v0*/, 0, mask_ty));

      if (vma == 0) { // undisturbed, read it first
         IRExpr* origin = getVReg(vd, 0, ty);
         IRExpr* inactive = binop(opofVecIR(vl, Iop_VAnd_vv_8 + index),
                                  unop(opofVecIR(vl, Iop_VNot_8 + index), mask),
                                  origin);
         IRExpr* active = binop(opofVecIR(vl, Iop_VAnd_vv_8 + index), mask, res);
         res = binop(opofVecIR(vl, Iop_VOr_vv_8 + index), active, inactive);
      } else { // agnostic, set to 1
         res = binop(opofVecIR(vl, Iop_VOr_vv_8 + index),
                     unop(opofVecIR(vl, Iop_VNot_8 + index), mask),
                     res);
      }
   }
   putVReg(irsb, vd, 0, res);

   return True;
}

static Bool dis_viota_m(/*MB_OUT*/ DisResult* dres,
                        /*OUT*/ IRSB*         irsb,
                        UInt                  insn,
                        Addr                  guest_pc_curr_instr,
                        VexGuestRISCV64State* guest)
{
   UInt vm = INSN(25, 25);
   UInt vs2 = INSN(24, 20);
   UInt vd = INSN(11, 7);

   UInt vma = SLICE_UInt(guest->guest_vtype, 7, 7);
   UInt sew = get_sew(guest);
   Int index = sewToIndex(sew);
   UInt vl = guest->guest_vl;
   IRType ty = typeofVecIR(vl, Ity_VLen8 + index);

   IRType mask_ty = typeofVecIR(vl, Ity_VLen1);
   IRExpr* e = getVReg(vs2, 0, mask_ty);
   if (vm == 0) {
      e = binop(opofVecIR(vl, Iop_VMand_mm), e, getVReg(0 /*v0*/, 0, mask_ty));
   }

   IRExpr* res = unop(opofVecIR(vl, Iop_VIota_m_8 + index), e);
   if (vm == 0) {
      IRExpr* mask = unop(opofVecIR(vl, Iop_VExpandBitsTo_8 + index),
                          getVReg(0 /*v0*/, 0, mask_ty));

      if (vma == 0) { // undisturbed, read it first
         IRExpr* origin = getVReg(vd, 0, ty);
         IRExpr* inactive = binop(opofVecIR(vl, Iop_VAnd_vv_8 + index),
                                  unop(opofVecIR(vl, Iop_VNot_8 + index), mask),
                                  origin);
         IRExpr* active = binop(opofVecIR(vl, Iop_VAnd_vv_8 + index), mask, res);
         res = binop(opofVecIR(vl, Iop_VOr_vv_8 + index), active, inactive);
      } else { // agnostic, set to 1
         res = binop(opofVecIR(vl, Iop_VOr_vv_8 + index),
                     unop(opofVecIR(vl, Iop_VNot_8 + index), mask),
                     res);
      }
   }
   putVReg(irsb, vd, 0, res);

   return True;
}

static Bool dis_vcompress_vm(/*MB_OUT*/ DisResult* dres,
                             /*OUT*/ IRSB*         irsb,
                             UInt                  insn,
                             Addr                  guest_pc_curr_instr,
                             VexGuestRISCV64State* guest)
{
   UInt vm = INSN(25, 25);
   UInt vs2 = INSN(24, 20);
   UInt vs1 = INSN(19, 15);
   UInt vd = INSN(11, 7);

   vassert(vm == 1);

   UInt sew = get_sew(guest);
   Int index = sewToIndex(sew);
   UInt vl = guest->guest_vl;
   IRType ty = typeofVecIR(vl, Ity_VLen8 + index);
   IRType mask_ty = typeofVecIR(vl, Ity_VLen1);

   putVReg(irsb, vd, 0, binop(opofVecIR(vl, Iop_VCompress_vm_8 + index),
                              getVReg(vs1, 0, mask_ty),
                              getVReg(vs2, 0, ty)));

   return True;
}

static Bool dis_opivv(/*MB_OUT*/ DisResult* dres,
                      /*OUT*/ IRSB*         irsb,
                      UInt                  insn,
                      Addr                  guest_pc_curr_instr,
                      VexGuestRISCV64State* guest)
{
   UInt funct6 = INSN(31, 26);
   UInt vm = INSN(25, 25);

   switch (funct6) {
   case 0b000000:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VAdd_vv_8);
   case 0b000010:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VSub_vv_8);
   case 0b000100:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMinu_vv_8);
   case 0b000101:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMin_vv_8);
   case 0b000110:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMaxu_vv_8);
   case 0b000111:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMax_vv_8);
   case 0b001010:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VOr_vv_8);
   case 0b001011:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VXor_vv_8);
   case 0b001001:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VAnd_vv_8);
   case 0b010000 ... 0b010011:
      return dis_rvv_addsub_carry(dres, irsb, insn, guest_pc_curr_instr, guest);
   case 0b010111:
      if (vm == 1) {
         return dis_rvv_vmv_v_vxi(dres, irsb, insn, guest_pc_curr_instr, guest, RVV_V);
      } else {
         return dis_rvv_vmerge_vxi(dres, irsb, insn, guest_pc_curr_instr, guest, RVV_V);
      }
      return False;
   case 0b011000:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMseq_vv_8);
   case 0b011001:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsne_vv_8);
   case 0b011010:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsltu_vv_8);
   case 0b011011:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMslt_vv_8);
   case 0b011100:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsleu_vv_8);
   case 0b011101:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsle_vv_8);
   case 0b100101:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VSll_vv_8);
   case 0b101000:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VSrl_vv_8);
   case 0b101001:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VSra_vv_8);
   case 0b101100:
      return dis_rvv2_w_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VNsrl_wv_8);
   case 0b101101:
      return dis_rvv2_w_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VNsra_wv_8);
   case 0b110000:
   case 0b110001:
      return dis_rvv_reduce(dres, irsb, insn, guest_pc_curr_instr, guest);

   default:
      return False;
   }
   return False;
}

static Bool dis_opmvv(/*MB_OUT*/ DisResult* dres,
                      /*OUT*/ IRSB*         irsb,
                      UInt                  insn,
                      Addr                  guest_pc_curr_instr,
                      VexGuestRISCV64State* guest)
{
   UInt funct6 = INSN(31, 26);

   switch (funct6) {
   case 0b000000 ...  0b000111:
      return dis_rvv_reduce(dres, irsb, insn, guest_pc_curr_instr, guest);
   case 0b010000:
      switch (INSN(19, 15)) {
      case 0b00000:
         return dis_rvv_vmv_x_s(dres, irsb, insn, guest_pc_curr_instr, guest);
      case 0b10000:
         return dis_vcpop_m(dres, irsb, insn, guest_pc_curr_instr, guest);
      case 0b10001:
         return dis_vfirst_m(dres, irsb, insn, guest_pc_curr_instr, guest);
      default:
         return False;
      }
      return False;
   case 0b010010:
      return dis_rvv_iext(dres, irsb, insn, guest_pc_curr_instr, guest);
   case 0b010100:
      switch (INSN(19, 15)) {
      case 0b00001:
         return dis_vmsBIOf_m(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsbf_m);
      case 0b00011:
         return dis_vmsBIOf_m(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsif_m);
      case 0b00010:
         return dis_vmsBIOf_m(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsof_m);
      case 0b10000:
         return dis_viota_m(dres, irsb, insn, guest_pc_curr_instr, guest);
      case 0b10001:
         return dis_vid_v(dres, irsb, insn, guest_pc_curr_instr, guest);
      default:
         return False;
      }
      return False;
   case 0b010111:
      return dis_vcompress_vm(dres, irsb, insn, guest_pc_curr_instr, guest);
   case 0b011001:
      return dis_rvv_m_m(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMand_mm);
   case 0b011101:
      return dis_rvv_m_m(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMnand_mm);
   case 0b011000:
      return dis_rvv_m_m(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMandn_mm);
   case 0b011011:
      return dis_rvv_m_m(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMxor_mm);
   case 0b011010:
      return dis_rvv_m_m(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMor_mm);
   case 0b011110:
      return dis_rvv_m_m(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMnor_mm);
   case 0b011100:
      return dis_rvv_m_m(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMorn_mm);
   case 0b011111:
      return dis_rvv_m_m(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMxnor_mm);
   case 0b100000:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VDivu_vv_8);
   case 0b100001:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VDiv_vv_8);
   case 0b100010:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VRemu_vv_8);
   case 0b100011:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VRem_vv_8);
   case 0b100101:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMul_vv_8);
   case 0b100111:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMulh_vv_8);
   case 0b100100:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMulhu_vv_8);
   case 0b100110:
      return dis_rvv2_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMulhsu_vv_8);
   case 0b101101:
      return dis_rvv3_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMacc_vv_8);
   case 0b101111:
      return dis_rvv3_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VNmsac_vv_8);
   case 0b101001:
      return dis_rvv3_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMadd_vv_8);
   case 0b101011:
      return dis_rvv3_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VNmsub_vv_8);
   case 0b110000:
      return dis_rvv2w_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWaddu_vv_8);
   case 0b110001:
      return dis_rvv2w_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWadd_vv_8);
   case 0b110010:
      return dis_rvv2w_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWsubu_vv_8);
   case 0b110011:
      return dis_rvv2w_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWsub_vv_8);
   case 0b110100:
      return dis_rvv2w_w_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWaddu_wv_8);
   case 0b110101:
      return dis_rvv2w_w_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWadd_wv_8);
   case 0b110110:
      return dis_rvv2w_w_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWsubu_wv_8);
   case 0b110111:
      return dis_rvv2w_w_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWsub_wv_8);
   case 0b111000:
      return dis_rvv2w_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWmulu_vv_8);
   case 0b111010:
      return dis_rvv2w_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWmulsu_vv_8);
   case 0b111011:
      return dis_rvv2w_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWmul_vv_8);
   case 0b111100:
      return dis_rvv3w_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWmaccu_vv_8);
   case 0b111101:
      return dis_rvv3w_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWmacc_vv_8);
   case 0b111111:
      return dis_rvv3w_v_v(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWmaccsu_vv_8);
   default:
      return False;
   }
   return False;
}

static Bool dis_opivi(/*MB_OUT*/ DisResult* dres,
                      /*OUT*/ IRSB*         irsb,
                      UInt                  insn,
                      Addr                  guest_pc_curr_instr,
                      VexGuestRISCV64State* guest)
{
   UInt funct6 = INSN(31, 26);
   UInt vm = INSN(25, 25);

   switch (funct6) {
   case 0b000000:
      return dis_rvv2_v_i(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VAdd_vi_8);
   case 0b000011:
      return dis_rvv2_v_i(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VRsub_vi_8);
   case 0b001010:
      return dis_rvv2_v_i(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VOr_vi_8);
   case 0b001011:
      return dis_rvv2_v_i(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VXor_vi_8);
   case 0b001001:
      return dis_rvv2_v_i(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VAnd_vi_8);
   case 0b010000 ... 0b010011:
      return dis_rvv_addsub_carry(dres, irsb, insn, guest_pc_curr_instr, guest);
   case 0b011000:
      return dis_rvv2_v_i(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMseq_vi_8);
   case 0b011001:
      return dis_rvv2_v_i(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsne_vi_8);
   case 0b011100:
      return dis_rvv2_v_i(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsleu_vi_8);
   case 0b011101:
      return dis_rvv2_v_i(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsle_vi_8);
   case 0b011110:
      return dis_rvv2_v_i(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsgtu_vi_8);
   case 0b011111:
      return dis_rvv2_v_i(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsgt_vi_8);
   case 0b010111:
      if (vm == 1) {
         return dis_rvv_vmv_v_vxi(dres, irsb, insn, guest_pc_curr_instr, guest, RVV_I);
      } else {
         return dis_rvv_vmerge_vxi(dres, irsb, insn, guest_pc_curr_instr, guest, RVV_I);
      }
      return False;
   case 0b100101:
      return dis_rvv2_v_i(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VSll_vi_8);
   case 0b100111:
      return dis_rvv_vmvr(dres, irsb, insn, guest_pc_curr_instr, guest);
   case 0b101000:
      return dis_rvv2_v_i(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VSrl_vi_8);
   case 0b101001:
      return dis_rvv2_v_i(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VSra_vi_8);
   case 0b101100:
      return dis_rvv2_w_i(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VNsrl_wi_8);
   case 0b101101:
      return dis_rvv2_w_i(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VNsra_wi_8);
   default:
      return False;
   }
}

static Bool dis_opivx(/*MB_OUT*/ DisResult* dres,
                      /*OUT*/ IRSB*         irsb,
                      UInt                  insn,
                      Addr                  guest_pc_curr_instr,
                      VexGuestRISCV64State* guest)
{
   UInt funct6 = INSN(31, 26);
   UInt vm = INSN(25, 25);

   switch (funct6) {
   case 0b000000:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VAdd_vx_8);
   case 0b000010:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VSub_vx_8);
   case 0b000011:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VRsub_vx_8);
   case 0b000100:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMinu_vx_8);
   case 0b000101:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMin_vx_8);
   case 0b000110:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMaxu_vx_8);
   case 0b000111:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMax_vx_8);
   case 0b001010:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VOr_vx_8);
   case 0b001011:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VXor_vx_8);
   case 0b001001:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VAnd_vx_8);
   case 0b010000 ... 0b010011:
      return dis_rvv_addsub_carry(dres, irsb, insn, guest_pc_curr_instr, guest);
   case 0b010111:
      if (vm == 1) {
         return dis_rvv_vmv_v_vxi(dres, irsb, insn, guest_pc_curr_instr, guest, RVV_X);
      } else {
         return dis_rvv_vmerge_vxi(dres, irsb, insn, guest_pc_curr_instr, guest, RVV_X);
      }
      return False;
   case 0b011000:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMseq_vx_8);
   case 0b011001:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsne_vx_8);
   case 0b011010:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsltu_vx_8);
   case 0b011011:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMslt_vx_8);
   case 0b011100:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsleu_vx_8);
   case 0b011101:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsle_vx_8);
   case 0b011110:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsgtu_vx_8);
   case 0b011111:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMsgt_vx_8);
   case 0b100101:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VSll_vx_8);
   case 0b101000:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VSrl_vx_8);
   case 0b101001:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VSra_vx_8);
   case 0b101100:
      return dis_rvv2_w_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VNsrl_wx_8);
   case 0b101101:
      return dis_rvv2_w_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VNsra_wx_8);
   default:
      return False;
   }
}

static Bool dis_opmvx(/*MB_OUT*/ DisResult* dres,
                      /*OUT*/ IRSB*         irsb,
                      UInt                  insn,
                      Addr                  guest_pc_curr_instr,
                      VexGuestRISCV64State* guest)
{
   UInt funct6 = INSN(31, 26);

   switch (funct6) {
   case 0b010000:
      switch (INSN(24, 20)) {
      case 0b00000:
         return dis_rvv_vmv_s_x(dres, irsb, insn, guest_pc_curr_instr, guest);
      default:
         return False;
      }
      return False;
   case 0b100000:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VDivu_vx_8);
   case 0b100001:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VDiv_vx_8);
   case 0b100010:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VRemu_vx_8);
   case 0b100011:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VRem_vx_8);
   case 0b100101:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMul_vx_8);
   case 0b100111:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMulh_vx_8);
   case 0b100100:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMulhu_vx_8);
   case 0b100110:
      return dis_rvv2_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMulhsu_vx_8);
   case 0b101101:
      return dis_rvv3_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMacc_vx_8);
   case 0b101111:
      return dis_rvv3_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VNmsac_vx_8);
   case 0b101001:
      return dis_rvv3_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VMadd_vx_8);
   case 0b101011:
      return dis_rvv3_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VNmsub_vx_8);
   case 0b110000:
      return dis_rvv2w_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWaddu_vx_8);
   case 0b110001:
      return dis_rvv2w_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWadd_vx_8);
   case 0b110010:
      return dis_rvv2w_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWsubu_vx_8);
   case 0b110011:
      return dis_rvv2w_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWsub_vx_8);
   case 0b110100:
      return dis_rvv2w_w_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWaddu_wx_8);
   case 0b110101:
      return dis_rvv2w_w_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWadd_wx_8);
   case 0b110110:
      return dis_rvv2w_w_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWsubu_wx_8);
   case 0b110111:
      return dis_rvv2w_w_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWsub_wx_8);
   case 0b111000:
      return dis_rvv2w_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWmulu_vx_8);
   case 0b111010:
      return dis_rvv2w_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWmulsu_vx_8);
   case 0b111011:
      return dis_rvv2w_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWmul_vx_8);
   case 0b111100:
      return dis_rvv3w_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWmaccu_vx_8);
   case 0b111101:
      return dis_rvv3w_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWmacc_vx_8);
   case 0b111111:
      return dis_rvv3w_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWmaccsu_vx_8);
   case 0b111110:
      return dis_rvv3w_v_x(dres, irsb, insn, guest_pc_curr_instr, guest, Iop_VWmaccus_vx_8);
   default:
      return False;
   }
   return False;
}

static UInt decode_eew(UInt raw_eew)
{
   switch (raw_eew) {
   case 0b000: return 8;
   case 0b101: return 16;
   case 0b110: return 32;
   case 0b111: return 64;
   default: vassert(0);
   }
}

static Bool dis_ldst(/*MB_OUT*/ DisResult* dres,
                     /*OUT*/ IRSB*         irsb,
                     UInt                  insn,
                     Addr                  guest_pc_curr_instr,
                     VexGuestRISCV64State* guest)
{
   UInt vd = INSN(11, 7);
   UInt width = INSN(14, 12);
   UInt rs1 = INSN(19, 15);
   UInt umop = INSN(24, 20);
   UInt vm = INSN(25, 25);
   UInt mew_mop = INSN(28, 26);

   // TODO: only part of  all ld/st instructions are handled
   if (!(mew_mop == 0b000 &&
         (umop == 0b00000 || umop == 0b10000))) {  // ignore fault-only-first
      return False;
   }

   Bool is_load = INSN(6, 0) == 0b0000111;

   DIP("%s - vl: %llu, insn: %x, vtype: %llx, vreg: %s\n",
       is_load ? "vload" : "vstore",
       guest->guest_vl, insn, guest->guest_vtype, nameVReg(vd));

   UInt eew_b = decode_eew(width) / 8;
   IRType ty = integerIRTypeOfSize(eew_b);
   UInt offset = 0;
   if (vm == 1) {  // disabled
      // It's possible to use larger ty them elem size
      for (UInt i = 0; i < guest->guest_vl; ++i) {
         IRExpr* addr = binop(Iop_Add64, getIReg64(rs1), mkU64(offset));

         if (is_load) {
            putVReg(irsb, vd, offset, loadLE(ty, addr));
         } else {
            storeLE(irsb, addr, getVReg(vd, offset, ty));
         }

         offset += eew_b;
      }
   } else {  // enabled
      for (UInt i = 0; i < guest->guest_vl; ++i) {
         IRExpr* addr = binop(Iop_Add64, getIReg64(rs1), mkU64(offset));
         UInt mask_offset = i / 64 * 8;
         IRExpr* guard = binop(Iop_CmpNE64,
                               mkU64(0),
                               binop(Iop_And64,
                                     getVReg(0 /* v0 */, mask_offset, Ity_I64),
                                     mkU64(1UL << (i % 64))));

         if (is_load) {
            IRLoadGOp no_cvt = ILGop_INVALID;
            switch (ty) {
               case Ity_I8:  no_cvt = ILGop_Ident8;  break;
               case Ity_I16: no_cvt = ILGop_Ident16;  break;
               case Ity_I32: no_cvt = ILGop_Ident32;  break;
               case Ity_I64: no_cvt = ILGop_Ident64;  break;
               default:  vassert(0);
            }

            UInt vma = SLICE_UInt(guest->guest_vtype, 7, 7);
            IRExpr* alt = (vma == 0) ? getVReg(vd, offset, ty) : mkU(ty, -1UL);
            IRTemp res = newTemp(irsb, ty);
            stmt(irsb, IRStmt_LoadG(Iend_LE, no_cvt, res, addr, alt, guard));
            putVReg(irsb, vd, offset, mkexpr(res));
         } else {
            stmt(irsb, IRStmt_StoreG(Iend_LE, addr, getVReg(vd, offset, ty), guard));
         }

         offset += eew_b;
      }
   }

   putPC(irsb, mkU64(guest_pc_curr_instr + 4));
   dres->whatNext    = Dis_StopHere;
   dres->jk_StopHere = Ijk_TooManyIR;

   return True;
}

static Bool dis_RV64V(/*MB_OUT*/ DisResult* dres,
                      /*OUT*/ IRSB*         irsb,
                      UInt                  insn,
                      Addr                  guest_pc_curr_instr,
                      const VexAbiInfo*     abiinfo)

{
   VexGuestRISCV64State* guest = abiinfo->riscv64_guest_state;

   // spec - 10. Vector Arithmetic Instruction Formats
   switch (INSN(6, 0)) {
   case 0b1010111:
      switch (INSN(14, 12)) {
      case 0b000:  // OPIVV
         return dis_opivv(dres, irsb, insn, guest_pc_curr_instr, guest);
      case 0b010:  // OPMVV
         return dis_opmvv(dres, irsb, insn, guest_pc_curr_instr, guest);
      case 0b011:  // OPIVI
         return dis_opivi(dres, irsb, insn, guest_pc_curr_instr, guest);
      case 0b100:  // OPIVX
         return dis_opivx(dres, irsb, insn, guest_pc_curr_instr, guest);
      case 0b110:  // OPMVX
         return dis_opmvx(dres, irsb, insn, guest_pc_curr_instr, guest);
      case 0b111:  // vsetvl
         return dis_vsetvl(dres, irsb, insn, guest_pc_curr_instr);
      default:
         return False;
      }
   case 0b0000111:  // load
   case 0b0100111:  // store
      return dis_ldst(dres, irsb, insn, guest_pc_curr_instr, guest);
   default:
      return False;
   }

   return False;
}

static Bool dis_RISCV64_standard(/*MB_OUT*/ DisResult* dres,
                                 /*OUT*/ IRSB*         irsb,
                                 UInt                  insn,
                                 Addr                  guest_pc_curr_instr,
                                 const VexAbiInfo*     abiinfo,
                                 Bool                  sigill_diag)
{
   vassert(INSN(1, 0) == 0b11);

   Bool ok = False;
   if (!ok)
      ok = dis_RV64I(dres, irsb, insn, guest_pc_curr_instr);
   if (!ok)
      ok = dis_RV64M(dres, irsb, insn);
   if (!ok)
      ok = dis_RV64A(dres, irsb, insn, guest_pc_curr_instr, abiinfo);
   if (!ok)
      ok = dis_RV64F(dres, irsb, insn);
   if (!ok)
      ok = dis_RV64D(dres, irsb, insn);
   if (!ok)
      ok = dis_RV64Zicsr(dres, irsb, insn);
   if (!ok)
      ok = dis_RV64V(dres, irsb, insn, guest_pc_curr_instr, abiinfo);
   if (ok)
      return True;

   if (sigill_diag)
      vex_printf("RISCV64 front end: standard\n");
   return False;
}

/* Disassemble a single riscv64 instruction into IR. Returns True iff the
   instruction was decoded, in which case *dres will be set accordingly, or
   False, in which case *dres should be ignored by the caller. */
static Bool disInstr_RISCV64_WRK(/*MB_OUT*/ DisResult* dres,
                                 /*OUT*/ IRSB*         irsb,
                                 const UChar*          guest_instr,
                                 Addr                  guest_pc_curr_instr,
                                 const VexArchInfo*    archinfo,
                                 const VexAbiInfo*     abiinfo,
                                 Bool                  sigill_diag)
{
   /* Set result defaults. */
   dres->whatNext    = Dis_Continue;
   dres->len         = 0;
   dres->jk_StopHere = Ijk_INVALID;
   dres->hint        = Dis_HintNone;

   /* Read the instruction word. */
   UInt insn = getInsn(guest_instr);

   if (0)
      vex_printf("insn: 0x%x\n", insn);

   DIP("\t(riscv64) 0x%llx:  ", (ULong)guest_pc_curr_instr);

   vassert((guest_pc_curr_instr & 1) == 0);

   /* Spot "Special" instructions (see comment at top of file). */
   {
      const UChar* code = guest_instr;
      /* Spot the 16-byte preamble:
            00305013   srli zero, zero, 3
            00d05013   srli zero, zero, 13
            03305013   srli zero, zero, 51
            03d05013   srli zero, zero, 61
      */
      UInt word1 = 0x00305013;
      UInt word2 = 0x00d05013;
      UInt word3 = 0x03305013;
      UInt word4 = 0x03d05013;
      if (getUIntLittleEndianly(code + 0) == word1 &&
          getUIntLittleEndianly(code + 4) == word2 &&
          getUIntLittleEndianly(code + 8) == word3 &&
          getUIntLittleEndianly(code + 12) == word4) {
         /* Got a "Special" instruction preamble. Which one is it? */
         dres->len  = 20;
         UInt which = getUIntLittleEndianly(code + 16);
         if (which == 0x00a56533 /* or a0, a0, a0 */) {
            /* a3 = client_request ( a4 ) */
            DIP("a3 = client_request ( a4 )\n");
            putPC(irsb, mkU64(guest_pc_curr_instr + 20));
            dres->jk_StopHere = Ijk_ClientReq;
            dres->whatNext    = Dis_StopHere;
            return True;
         } else if (which == 0x00b5e5b3 /* or a1, a1, a1 */) {
            /* a3 = guest_NRADDR */
            DIP("a3 = guest_NRADDR\n");
            putIReg64(irsb, 13 /*x13/a3*/, IRExpr_Get(OFFB_NRADDR, Ity_I64));
            return True;
         } else if (which == 0x00c66633 /* or a2, a2, a2 */) {
            /* branch-and-link-to-noredir t0 */
            DIP("branch-and-link-to-noredir t0\n");
            putIReg64(irsb, 1 /*x1/ra*/, mkU64(guest_pc_curr_instr + 20));
            putPC(irsb, getIReg64(5 /*x5/t0*/));
            dres->jk_StopHere = Ijk_NoRedir;
            dres->whatNext    = Dis_StopHere;
            return True;
         } else if (which == 0x00d6e6b3 /* or a3, a3, a3 */) {
            /* IR injection */
            DIP("IR injection\n");
            vex_inject_ir(irsb, Iend_LE);
            /* Invalidate the current insn. The reason is that the IRop we're
               injecting here can change. In which case the translation has to
               be redone. For ease of handling, we simply invalidate all the
               time. */
            stmt(irsb, IRStmt_Put(OFFB_CMSTART, mkU64(guest_pc_curr_instr)));
            stmt(irsb, IRStmt_Put(OFFB_CMLEN, mkU64(20)));
            putPC(irsb, mkU64(guest_pc_curr_instr + 20));
            dres->whatNext    = Dis_StopHere;
            dres->jk_StopHere = Ijk_InvalICache;
            return True;
         }
         /* We don't know what it is. */
         return False;
      }
   }

   /* Main riscv64 instruction decoder starts here. */
   Bool ok = False;
   UInt inst_size;

   /* Parse insn[1:0] to determine whether the instruction is 16-bit
      (compressed) or 32-bit. */
   switch (INSN(1, 0)) {
   case 0b00:
   case 0b01:
   case 0b10:
      dres->len = inst_size = 2;
      ok = dis_RV64C(dres, irsb, insn, guest_pc_curr_instr, sigill_diag);
      break;

   case 0b11:
      dres->len = inst_size = 4;
      ok = dis_RISCV64_standard(dres, irsb, insn, guest_pc_curr_instr, abiinfo,
                                sigill_diag);
      break;

   default:
      vassert(0); /* Can't happen. */
   }

   /* If the next-level down decoders failed, make sure dres didn't get
      changed. */
   if (!ok) {
      vassert(dres->whatNext == Dis_Continue);
      vassert(dres->len == inst_size);
      vassert(dres->jk_StopHere == Ijk_INVALID);
   }

   return ok;
}

#undef INSN

/*------------------------------------------------------------*/
/*--- Top-level fn                                         ---*/
/*------------------------------------------------------------*/

/* Disassemble a single instruction into IR. The instruction is located in host
   memory at &guest_code[delta]. */
DisResult disInstr_RISCV64(IRSB*              irsb,
                           const UChar*       guest_code,
                           Long               delta,
                           Addr               guest_IP,
                           VexArch            guest_arch,
                           const VexArchInfo* archinfo,
                           const VexAbiInfo*  abiinfo,
                           VexEndness         host_endness,
                           Bool               sigill_diag)
{
   DisResult dres;
   vex_bzero(&dres, sizeof(dres));

   vassert(guest_arch == VexArchRISCV64);
   /* Check that the host is little-endian as getFReg32() and putFReg32() depend
      on this fact. */
   vassert(host_endness == VexEndnessLE);

   /* Try to decode. */
   Bool ok = disInstr_RISCV64_WRK(&dres, irsb, &guest_code[delta], guest_IP,
                                  archinfo, abiinfo, sigill_diag);
   if (ok) {
      /* All decode successes end up here. */
      vassert(dres.len == 2 || dres.len == 4 || dres.len == 20);
      switch (dres.whatNext) {
      case Dis_Continue:
         putPC(irsb, mkU64(guest_IP + dres.len));
         break;
      case Dis_StopHere:
         break;
      default:
         vassert(0);
      }
      DIP("\n");
   } else {
      /* All decode failures end up here. */
      if (sigill_diag) {
         Int   i, j;
         UChar buf[64];
         UInt  insn = getInsn(&guest_code[delta]);
         vex_bzero(buf, sizeof(buf));
         for (i = j = 0; i < 32; i++) {
            if (i > 0) {
               if ((i & 7) == 0)
                  buf[j++] = ' ';
               else if ((i & 3) == 0)
                  buf[j++] = '\'';
            }
            buf[j++] = (insn & (1 << (31 - i))) ? '1' : '0';
         }
         vex_printf("disInstr(riscv64): unhandled instruction 0x%08x\n", insn);
         vex_printf("disInstr(riscv64): %s\n", buf);
      }

      /* Tell the dispatcher that this insn cannot be decoded, and so has not
         been executed, and (is currently) the next to be executed. The pc
         register should be up-to-date since it is made so at the start of each
         insn, but nevertheless be paranoid and update it again right now. */
      putPC(irsb, mkU64(guest_IP));
      dres.len         = 0;
      dres.whatNext    = Dis_StopHere;
      dres.jk_StopHere = Ijk_NoDecode;
   }
   return dres;
}

/*--------------------------------------------------------------------*/
/*--- end                                     guest_riscv64_toIR.c ---*/
/*--------------------------------------------------------------------*/
