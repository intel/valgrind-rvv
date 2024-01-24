
/*--------------------------------------------------------------------*/
/*--- begin                                    host_riscv64_isel.c ---*/
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

#include "host_riscv64_defs.h"
#include "main_globals.h"
#include "main_util.h"

/*------------------------------------------------------------*/
/*--- ISelEnv                                              ---*/
/*------------------------------------------------------------*/

/* This carries around:

   - A mapping from IRTemp to IRType, giving the type of any IRTemp we might
     encounter. This is computed before insn selection starts, and does not
     change.

   - A mapping from IRTemp to HReg. This tells the insn selector which virtual
     register is associated with each IRTemp temporary. This is computed before
     insn selection starts, and does not change. We expect this mapping to map
     precisely the same set of IRTemps as the type mapping does.

     - vregmap   holds the primary register for the IRTemp.
     - vregmapHI is only used for 128-bit integer-typed IRTemps. It holds the
                 identity of a second 64-bit virtual HReg, which holds the high
                 half of the value.

   - The code array, that is, the insns selected so far.

   - A counter, for generating new virtual registers.

   - The host hardware capabilities word. This is set at the start and does not
     change.

   - A Bool for indicating whether we may generate chain-me instructions for
     control flow transfers, or whether we must use XAssisted.

   - The maximum guest address of any guest insn in this block. Actually, the
     address of the highest-addressed byte from any insn in this block. Is set
     at the start and does not change. This is used for detecting jumps which
     are definitely forward-edges from this block, and therefore can be made
     (chained) to the fast entry point of the destination, thereby avoiding the
     destination's event check.

   - An IRExpr*, which may be NULL, holding the IR expression (an
     IRRoundingMode-encoded value) to which the FPU's rounding mode was most
     recently set. Setting to NULL is always safe. Used to avoid redundant
     settings of the FPU's rounding mode, as described in
     set_fcsr_rounding_mode() below.

   Note, this is all (well, mostly) host-independent.
*/

typedef struct {
   UInt vlmax;
   UInt padding;
} VecEnv;

typedef struct {
   /* Constant -- are set at the start and do not change. */
   IRTypeEnv* type_env;
   VecEnv vec_env;

   HReg* vregmaps[8];  // TODO
   HReg* vregmapHI;
   Int   n_vregmap;

   UInt hwcaps;

   Bool   chainingAllowed;
   Addr64 max_ga;

   /* These are modified as we go along. */
   HInstrArray* code;
   Int          vreg_ctr;

   IRExpr* previous_rm;
} ISelEnv;

#define vregmap vregmaps[0]
#define vregmapHI vregmaps[1]

#define MAX(a, b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define MIN(a, b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#define DIV_ROUND_UP(a, b)  (((a) + (b) - 1) / (b))
#define ROUND_UP(a, b)  (((a) + (b) - 1) / (b) * (b))

static HReg lookupIRTemp(ISelEnv* env, IRTemp tmp)
{
   vassert(tmp >= 0);
   vassert(tmp < env->n_vregmap);
   return env->vregmap[tmp];
}

static void lookupIRTempVec(HReg vec[], Int nregs, ISelEnv* env, IRTemp tmp)
{
   vassert(tmp >= 0);
   vassert(tmp < env->n_vregmap);
   for (int i = 0; i < nregs; ++i) {
      vec[i] = env->vregmaps[i][tmp];
   }
}

static void addInstr(ISelEnv* env, RISCV64Instr* instr)
{
   addHInstr(env->code, instr);
   if (vex_traceflags & VEX_TRACE_VCODE) {
      ppRISCV64Instr(instr, True /*mode64*/);
      vex_printf("\n");
   }
}

static HReg newVRegI(ISelEnv* env)
{
   HReg reg = mkHReg(True /*virtual*/, HRcInt64, 0, env->vreg_ctr);
   env->vreg_ctr++;
   return reg;
}

static HReg newVRegF(ISelEnv* env)
{
   HReg reg = mkHReg(True /*virtual*/, HRcFlt64, 0, env->vreg_ctr);
   env->vreg_ctr++;
   return reg;
}

static HReg newVRegV(ISelEnv* env)
{
   HReg reg = mkHReg(True /*virtual*/, HRcVecVLen, 0, env->vreg_ctr);
   env->vreg_ctr++;
   return reg;
}
/*------------------------------------------------------------*/
/*--- ISEL: Forward declarations                           ---*/
/*------------------------------------------------------------*/

/* These are organised as iselXXX and iselXXX_wrk pairs. The iselXXX_wrk do the
   real work, but are not to be called directly. For each XXX, iselXXX calls its
   iselXXX_wrk counterpart, then checks that all returned registers are virtual.
   You should not call the _wrk version directly. */

static HReg iselIntExpr_R(ISelEnv* env, IRExpr* e);
static void iselInt128Expr(HReg* rHi, HReg* rLo, ISelEnv* env, IRExpr* e);
static HReg iselFltExpr(ISelEnv* env, IRExpr* e);
static void iselVecExpr_R(HReg r[], ISelEnv* env, IRExpr* e);

/*------------------------------------------------------------*/
/*--- ISEL: FP rounding mode helpers                       ---*/
/*------------------------------------------------------------*/

/* Set the FP rounding mode: 'mode' is an I32-typed expression denoting a value
   of IRRoundingMode. Set the fcsr RISC-V register to have the same rounding.

   All attempts to set the rounding mode have to be routed through this
   function for things to work properly. Refer to the comment in the AArch64
   backend for set_FPCR_rounding_mode() how the mechanism relies on the SSA
   property of IR and CSE.
*/
static void set_fcsr_rounding_mode(ISelEnv* env, IRExpr* mode)
{
   vassert(typeOfIRExpr(env->type_env, mode) == Ity_I32);

   /* Do we need to do anything? */
   if (env->previous_rm && env->previous_rm->tag == Iex_RdTmp &&
       mode->tag == Iex_RdTmp &&
       env->previous_rm->Iex.RdTmp.tmp == mode->Iex.RdTmp.tmp) {
      /* No - setting it to what it was before.  */
      vassert(typeOfIRExpr(env->type_env, env->previous_rm) == Ity_I32);
      return;
   }

   /* No luck - we better set it, and remember what we set it to. */
   env->previous_rm = mode;

   /*
      rounding mode                 |  IR  | RISC-V
      ---------------------------------------------
      to nearest, ties to even      | 0000 |   000
      to -infinity                  | 0001 |   011
      to +infinity                  | 0010 |   010
      to zero                       | 0011 |   001
      to nearest, ties away from 0  | 0100 |   100
      prepare for shorter precision | 0101 |   111
      to away from 0                | 0110 |   111
      to nearest, ties towards 0    | 0111 |   111
      invalid                       | 1000 |   111

      All rounding modes not supported on RISC-V are mapped to 111 which is the
      dynamic mode that is always invalid in fcsr and raises an illegal
      instruction exception.

      The mapping can be implemented using the following transformation:
         t0 = 30 >> rm_IR
         t1 = t0 & 19
         t2 = t0 + 7
         t3 = t1 + t2
         fcsr_rm_RISCV = t3 >> t1
   */
   HReg rm_IR  = iselIntExpr_R(env, mode);
   HReg imm_30 = newVRegI(env);
   addInstr(env, RISCV64Instr_LI(imm_30, 30));
   HReg t0 = newVRegI(env);
   addInstr(env, RISCV64Instr_ALU(RISCV64op_SRL, t0, imm_30, rm_IR));
   HReg t1 = newVRegI(env);
   addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ANDI, t1, t0, 19));
   HReg t2 = newVRegI(env);
   addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, t2, t0, 7));
   HReg t3 = newVRegI(env);
   addInstr(env, RISCV64Instr_ALU(RISCV64op_ADD, t3, t1, t2));
   HReg fcsr_rm_RISCV = newVRegI(env);
   addInstr(env, RISCV64Instr_ALU(RISCV64op_SRL, fcsr_rm_RISCV, t3, t1));
   addInstr(env,
            RISCV64Instr_CSRRW(hregRISCV64_x0(), fcsr_rm_RISCV, 0x002 /*frm*/));
}

/*------------------------------------------------------------*/
/*--- ISEL: Function call helpers                          ---*/
/*------------------------------------------------------------*/

/* Used only in doHelperCall(). See the big comment in doHelperCall() regarding
   handling of register-parameter arguments. This function figures out whether
   evaluation of an expression might require use of a fixed register. If in
   doubt return True (safe but suboptimal).
*/
static Bool mightRequireFixedRegs(IRExpr* e)
{
   if (UNLIKELY(is_IRExpr_VECRET_or_GSPTR(e))) {
      /* These are always "safe" -- either a copy of x2/sp in some arbitrary
         vreg, or a copy of x8/s0, respectively. */
      return False;
   }
   /* Else it's a "normal" expression. */
   switch (e->tag) {
   case Iex_RdTmp:
   case Iex_Const:
   case Iex_Get:
      return False;
   default:
      return True;
   }
}

/* Do a complete function call. |guard| is a Ity_Bit expression indicating
   whether or not the call happens. If guard==NULL, the call is unconditional.
   |retloc| is set to indicate where the return value is after the call. The
   caller (of this fn) must generate code to add |stackAdjustAfterCall| to the
   stack pointer after the call is done. Returns True iff it managed to handle
   this combination of arg/return types, else returns False. */
static Bool doHelperCall(/*OUT*/ UInt*   stackAdjustAfterCall,
                         /*OUT*/ RetLoc* retloc,
                         ISelEnv*        env,
                         IRExpr*         guard,
                         IRCallee*       cee,
                         IRType          retTy,
                         IRExpr**        args)
{
   /* Set default returns. We'll update them later if needed. */
   *stackAdjustAfterCall = 0;
   *retloc               = mk_RetLoc_INVALID();

   /* Marshal args for a call and do the call.

      This function only deals with a limited set of possibilities, which cover
      all helpers in practice. The restrictions are that only the following
      arguments are supported:
      * RISCV64_N_REGPARMS x Ity_I32/Ity_I64 values, passed in x10/a0 .. x17/a7,
      * RISCV64_N_FREGPARMS x Ity_F32/Ity_F64 values, passed in f10/fa0 ..
        f17/fa7.

      Note that the cee->regparms field is meaningless on riscv64 hosts (since
      we only implement one calling convention) and so we always ignore it.

      The return type can be I{8,16,32,64} or V128. In the V128 case, it is
      expected that |args| will contain the special node IRExpr_VECRET(), in
      which case this routine generates code to allocate space on the stack for
      the vector return value.  Since we are not passing any scalars on the
      stack, it is enough to preallocate the return space before marshalling any
      arguments, in this case.

      |args| may also contain IRExpr_GSPTR(), in which case the value in the
      guest state pointer register minus BASEBLOCK_OFFSET_ADJUSTMENT is passed
      as the corresponding argument.

      Generating code which is both efficient and correct when parameters are to
      be passed in registers is difficult, for the reasons elaborated in detail
      in comments attached to doHelperCall() in VEX/priv/host_x86_isel.c. Here,
      we use a variant of the method described in those comments.

      The problem is split into two cases: the fast scheme and the slow scheme.
      In the fast scheme, arguments are computed directly into the target (real)
      registers. This is only safe when we can be sure that computation of each
      argument will not trash any real registers set by computation of any other
      argument.

      In the slow scheme, all args are first computed into vregs, and once they
      are all done, they are moved to the relevant real regs. This always gives
      correct code, but it also gives a bunch of vreg-to-rreg moves which are
      usually redundant but are hard for the register allocator to get rid of.

      To decide which scheme to use, all argument expressions are first
      examined. If they are all so simple that it is clear they will be
      evaluated without use of any fixed registers, use the fast scheme, else
      use the slow scheme. Note also that only unconditional calls may use the
      fast scheme, since having to compute a condition expression could itself
      trash real registers.

      Note this requires being able to examine an expression and determine
      whether or not evaluation of it might use a fixed register. That requires
      knowledge of how the rest of this insn selector works. Currently just the
      following 3 are regarded as safe -- hopefully they cover the majority of
      arguments in practice: IRExpr_RdTmp, IRExpr_Const, IRExpr_Get.
   */

   /* These are used for cross-checking that IR-level constraints on the use of
      IRExpr_VECRET() and IRExpr_GSPTR() are observed. */
   UInt nVECRETs = 0;
   UInt nGSPTRs  = 0;

   UInt n_args = 0;
   for (UInt i = 0; args[i] != NULL; i++) {
      IRExpr* arg = args[i];
      if (UNLIKELY(arg->tag == Iex_VECRET))
         nVECRETs++;
      else if (UNLIKELY(arg->tag == Iex_GSPTR))
         nGSPTRs++;
      n_args++;
   }

   /* If this fails, the IR is ill-formed. */
   vassert(nGSPTRs == 0 || nGSPTRs == 1);

   /* If we have a VECRET, allocate space on the stack for the return value, and
      record the stack pointer after that. */
   HReg r_vecRetAddr = INVALID_HREG;
   if (nVECRETs == 1) {
      vassert(retTy == Ity_V128 || retTy == Ity_V256);
      r_vecRetAddr = newVRegI(env);
      addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, hregRISCV64_x2(),
                                        hregRISCV64_x2(),
                                        retTy == Ity_V128 ? -16 : -32));
      addInstr(env, RISCV64Instr_MV(r_vecRetAddr, hregRISCV64_x2()));
   } else {
      /* If either of these fail, the IR is ill-formed. */
      vassert(retTy != Ity_V128 && retTy != Ity_V256);
      vassert(nVECRETs == 0);
   }

   /* First decide which scheme (slow or fast) is to be used. First assume the
      fast scheme, and select slow if any contraindications (wow) appear. */
   Bool go_fast = True;

   /* We'll need space on the stack for the return value. Avoid possible
      complications with nested calls by using the slow scheme. */
   if (retTy == Ity_V128 || retTy == Ity_V256)
      go_fast = False;

   if (go_fast && guard != NULL) {
      if (guard->tag == Iex_Const && guard->Iex.Const.con->tag == Ico_U1 &&
          guard->Iex.Const.con->Ico.U1 == True) {
         /* Unconditional. */
      } else {
         /* Not manifestly unconditional -- be conservative. */
         go_fast = False;
      }
   }

   if (go_fast)
      for (UInt i = 0; i < n_args; i++) {
         if (mightRequireFixedRegs(args[i])) {
            go_fast = False;
            break;
         }
      }

   /* At this point the scheme to use has been established. Generate code to get
      the arg values into the argument regs. If we run out of arg regs, give up.
    */

   HReg argregs[RISCV64_N_ARGREGS];
   HReg fargregs[RISCV64_N_FARGREGS];

   vassert(RISCV64_N_ARGREGS == 8);
   vassert(RISCV64_N_FARGREGS == 8);

   argregs[0] = hregRISCV64_x10();
   argregs[1] = hregRISCV64_x11();
   argregs[2] = hregRISCV64_x12();
   argregs[3] = hregRISCV64_x13();
   argregs[4] = hregRISCV64_x14();
   argregs[5] = hregRISCV64_x15();
   argregs[6] = hregRISCV64_x16();
   argregs[7] = hregRISCV64_x17();

   fargregs[0] = hregRISCV64_f10();
   fargregs[1] = hregRISCV64_f11();
   fargregs[2] = hregRISCV64_f12();
   fargregs[3] = hregRISCV64_f13();
   fargregs[4] = hregRISCV64_f14();
   fargregs[5] = hregRISCV64_f15();
   fargregs[6] = hregRISCV64_f16();
   fargregs[7] = hregRISCV64_f17();

   HReg tmpregs[RISCV64_N_ARGREGS];
   HReg ftmpregs[RISCV64_N_FARGREGS];
   Int  nextArgReg = 0, nextFArgReg = 0;
   HReg cond;

   if (go_fast) {
      /* FAST SCHEME */
      for (UInt i = 0; i < n_args; i++) {
         IRExpr* arg = args[i];

         IRType aTy = Ity_INVALID;
         if (LIKELY(!is_IRExpr_VECRET_or_GSPTR(arg)))
            aTy = typeOfIRExpr(env->type_env, args[i]);

         if (aTy == Ity_I32 || aTy == Ity_I64) {
            if (nextArgReg >= RISCV64_N_ARGREGS)
               return False; /* Out of argregs. */
            addInstr(env, RISCV64Instr_MV(argregs[nextArgReg],
                                          iselIntExpr_R(env, args[i])));
            nextArgReg++;
         } else if (aTy == Ity_F32 || aTy == Ity_F64) {
            if (nextFArgReg >= RISCV64_N_FARGREGS)
               return False; /* Out of fargregs. */
            addInstr(env,
                     RISCV64Instr_FpMove(RISCV64op_FMV_D, fargregs[nextFArgReg],
                                         iselFltExpr(env, args[i])));
            nextFArgReg++;
         } else if (arg->tag == Iex_GSPTR) {
            if (nextArgReg >= RISCV64_N_ARGREGS)
               return False; /* Out of argregs. */
	    /* See dispatch-riscv64-linux.S for -2048 */
            addInstr(env,
                     RISCV64Instr_ALUImm(RISCV64op_ADDI, argregs[nextArgReg],
                                         hregRISCV64_x8(), -2048));
            nextArgReg++;
         } else if (arg->tag == Iex_VECRET) {
            /* Because of the go_fast logic above, we can't get here, since
               vector return values make us use the slow path instead. */
            vassert(0);
         } else
            return False; /* Unhandled arg type. */
      }

      /* Fast scheme only applies for unconditional calls. Hence: */
      cond = INVALID_HREG;

   } else {
      /* SLOW SCHEME; move via temporaries. */
      for (UInt i = 0; i < n_args; i++) {
         IRExpr* arg = args[i];

         IRType aTy = Ity_INVALID;
         if (LIKELY(!is_IRExpr_VECRET_or_GSPTR(arg)))
            aTy = typeOfIRExpr(env->type_env, args[i]);

         if (aTy == Ity_I32 || aTy == Ity_I64) {
            if (nextArgReg >= RISCV64_N_ARGREGS)
               return False; /* Out of argregs. */
            tmpregs[nextArgReg] = iselIntExpr_R(env, args[i]);
            nextArgReg++;
         } else if (aTy == Ity_F32 || aTy == Ity_F64) {
            if (nextFArgReg >= RISCV64_N_FARGREGS)
               return False; /* Out of fargregs. */
            ftmpregs[nextFArgReg] = iselFltExpr(env, args[i]);
            nextFArgReg++;
         } else if (arg->tag == Iex_GSPTR) {
            if (nextArgReg >= RISCV64_N_ARGREGS)
               return False; /* Out of argregs. */

            addInstr(env,
                     RISCV64Instr_ALUImm(RISCV64op_ADDI, tmpregs[nextArgReg],
                                         hregRISCV64_x8(), -2048));
            nextArgReg++;
         } else if (arg->tag == Iex_VECRET) {
            vassert(!hregIsInvalid(r_vecRetAddr));
            tmpregs[nextArgReg] = r_vecRetAddr;
            nextArgReg++;
         } else
            return False; /* Unhandled arg type. */
      }

      /* Compute the condition. Be a bit clever to handle the common case where
         the guard is 1:Bit. */
      cond = INVALID_HREG;
      if (guard) {
         if (guard->tag == Iex_Const && guard->Iex.Const.con->tag == Ico_U1 &&
             guard->Iex.Const.con->Ico.U1 == True) {
            /* Unconditional -- do nothing. */
         } else {
            cond = iselIntExpr_R(env, guard);
         }
      }

      /* Move the args to their final destinations. */
      for (UInt i = 0; i < nextArgReg; i++) {
         vassert(!(hregIsInvalid(tmpregs[i])));
         addInstr(env, RISCV64Instr_MV(argregs[i], tmpregs[i]));
      }
      for (UInt i = 0; i < nextFArgReg; i++) {
         vassert(!(hregIsInvalid(ftmpregs[i])));
         addInstr(env, RISCV64Instr_FpMove(RISCV64op_FMV_D, fargregs[i],
                                           ftmpregs[i]));
      }
   }

   /* Should be assured by checks above. */
   vassert(nextArgReg <= RISCV64_N_ARGREGS);
   vassert(nextFArgReg <= RISCV64_N_FARGREGS);

   /* Do final checks, set the return values, and generate the call instruction
      proper. */
   vassert(nGSPTRs == 0 || nGSPTRs == 1);
   vassert(nVECRETs == ((retTy == Ity_V128 || retTy == Ity_V256) ? 1 : 0));
   vassert(*stackAdjustAfterCall == 0);
   vassert(is_RetLoc_INVALID(*retloc));
   switch (retTy) {
   case Ity_INVALID:
      /* Function doesn't return a value. */
      *retloc = mk_RetLoc_simple(RLPri_None);
      break;
   case Ity_I8:
   case Ity_I16:
   case Ity_I32:
   case Ity_I64:
      *retloc = mk_RetLoc_simple(RLPri_Int);
      break;
   case Ity_V128:
      *retloc               = mk_RetLoc_spRel(RLPri_V128SpRel, 0);
      *stackAdjustAfterCall = 16;
      break;
   case Ity_V256:
      *retloc               = mk_RetLoc_spRel(RLPri_V256SpRel, 0);
      *stackAdjustAfterCall = 32;
      break;
   default:
      /* IR can denote other possible return types, but we don't handle those
         here. */
      return False;
   }

   /* Finally, generate the call itself. This needs the *retloc value set in the
      switch above, which is why it's at the end. */

   /* nextArgReg doles out argument registers. Since these are assigned in the
      order x10/a0 .. x17/a7, its numeric value at this point, which must be
      between 0 and 8 inclusive, is going to be equal to the number of arg regs
      in use for the call. Hence bake that number into the call (we'll need to
      know it when doing register allocation, to know what regs the call reads.)

      The same applies to nextFArgReg which records a number of used
      floating-point registers f10/fa0 .. f17/fa7.
    */
   addInstr(env, RISCV64Instr_Call(*retloc, (Addr64)cee->addr, cond, nextArgReg,
                                   nextFArgReg));

   return True;
}

/*------------------------------------------------------------*/
/*--- ISEL: Integer expressions (64/32/16/8/1 bit)         ---*/
/*------------------------------------------------------------*/

/* Select insns for an integer-typed expression, and add them to the code list.
   Return a reg holding the result. This reg will be a virtual register. THE
   RETURNED REG MUST NOT BE MODIFIED. If you want to modify it, ask for a new
   vreg, copy it in there, and modify the copy. The register allocator will do
   its best to map both vregs to the same real register, so the copies will
   often disappear later in the game.

   This should handle expressions of 64, 32, 16, 8 and 1-bit type. All results
   are returned in a 64-bit register. For an N-bit expression, the upper 64-N
   bits are arbitrary, so you should mask or sign-extend partial values if
   necessary.

   The riscv64 backend however internally always extends the values as follows:
   * a 32/16/8-bit integer result is sign-extended to 64 bits,
   * a 1-bit logical result is zero-extended to 64 bits.

   This schema follows the approach taken by the RV64 ISA which by default
   sign-extends any 32/16/8-bit operation result to 64 bits. Matching the isel
   with the ISA generally results in requiring less instructions. For instance,
   it allows that any Ico_U32 immediate can be always materialized at maximum
   using two instructions (LUI+ADDIW).

   An important consequence of this design is that any Iop_<N>Sto64 extension is
   a no-op. On the other hand, any Iop_64to<N> operation must additionally
   perform an N-bit sign-extension. This is the opposite situation than in most
   other VEX backends.
*/

/* -------------------------- Reg --------------------------- */

/* DO NOT CALL THIS DIRECTLY ! */
static HReg iselIntExpr_R_wrk(ISelEnv* env, IRExpr* e)
{
   IRType ty = typeOfIRExpr(env->type_env, e);
   vassert(ty == Ity_I64 || ty == Ity_I32 || ty == Ity_I16 || ty == Ity_I8 ||
           ty == Ity_I1);

   switch (e->tag) {
   /* ------------------------ TEMP ------------------------- */
   case Iex_RdTmp: {
      return lookupIRTemp(env, e->Iex.RdTmp.tmp);
   }

   /* ------------------------ LOAD ------------------------- */
   case Iex_Load: {
      if (e->Iex.Load.end != Iend_LE)
         goto irreducible;

      HReg dst = newVRegI(env);
      /* TODO Optimize the cases with small imm Add64/Sub64. */
      HReg addr = iselIntExpr_R(env, e->Iex.Load.addr);

      if (ty == Ity_I64)
         addInstr(env, RISCV64Instr_Load(RISCV64op_LD, dst, addr, 0));
      else if (ty == Ity_I32)
         addInstr(env, RISCV64Instr_Load(RISCV64op_LW, dst, addr, 0));
      else if (ty == Ity_I16)
         addInstr(env, RISCV64Instr_Load(RISCV64op_LH, dst, addr, 0));
      else if (ty == Ity_I8)
         addInstr(env, RISCV64Instr_Load(RISCV64op_LB, dst, addr, 0));
      else
         goto irreducible;
      return dst;
   }

   /* ---------------------- BINARY OP ---------------------- */
   case Iex_Binop: {
      /* TODO Optimize for small imms by generating <instr>i. */
      switch (e->Iex.Binop.op) {
      case Iop_Add64:
      case Iop_Add32:
      case Iop_Sub64:
      case Iop_Sub32:
      case Iop_Xor64:
      case Iop_Xor32:
      case Iop_Or64:
      case Iop_Or32:
      case Iop_Or16:
      case Iop_Or8:
      case Iop_Or1:
      case Iop_And64:
      case Iop_And32:
      case Iop_And1:
      case Iop_Shl64:
      case Iop_Shl32:
      case Iop_Shr64:
      case Iop_Shr32:
      case Iop_Sar64:
      case Iop_Sar32:
      case Iop_Mul64:
      case Iop_Mul32:
      case Iop_DivU64:
      case Iop_DivU32:
      case Iop_DivS64:
      case Iop_DivS32: {
         RISCV64ALUOp op;
         switch (e->Iex.Binop.op) {
         case Iop_Add64:
            op = RISCV64op_ADD;
            break;
         case Iop_Add32:
            op = RISCV64op_ADDW;
            break;
         case Iop_Sub64:
            op = RISCV64op_SUB;
            break;
         case Iop_Sub32:
            op = RISCV64op_SUBW;
            break;
         case Iop_Xor64:
         case Iop_Xor32:
            op = RISCV64op_XOR;
            break;
         case Iop_Or64:
         case Iop_Or32:
         case Iop_Or16:
         case Iop_Or8:
         case Iop_Or1:
            op = RISCV64op_OR;
            break;
         case Iop_And64:
         case Iop_And32:
         case Iop_And1:
            op = RISCV64op_AND;
            break;
         case Iop_Shl64:
            op = RISCV64op_SLL;
            break;
         case Iop_Shl32:
            op = RISCV64op_SLLW;
            break;
         case Iop_Shr64:
            op = RISCV64op_SRL;
            break;
         case Iop_Shr32:
            op = RISCV64op_SRLW;
            break;
         case Iop_Sar64:
            op = RISCV64op_SRA;
            break;
         case Iop_Sar32:
            op = RISCV64op_SRAW;
            break;
         case Iop_Mul64:
            op = RISCV64op_MUL;
            break;
         case Iop_Mul32:
            op = RISCV64op_MULW;
            break;
         case Iop_DivU64:
            op = RISCV64op_DIVU;
            break;
         case Iop_DivU32:
            op = RISCV64op_DIVUW;
            break;
         case Iop_DivS64:
            op = RISCV64op_DIV;
            break;
         case Iop_DivS32:
            op = RISCV64op_DIVW;
            break;
         default:
            vassert(0);
         }
         HReg dst  = newVRegI(env);
         HReg argL = iselIntExpr_R(env, e->Iex.Binop.arg1);
         HReg argR = iselIntExpr_R(env, e->Iex.Binop.arg2);
         addInstr(env, RISCV64Instr_ALU(op, dst, argL, argR));
         return dst;
      }
      case Iop_CmpEQ64:
      case Iop_CmpEQ32:
      case Iop_CasCmpEQ64:
      case Iop_CasCmpEQ32: {
         HReg tmp  = newVRegI(env);
         HReg argL = iselIntExpr_R(env, e->Iex.Binop.arg1);
         HReg argR = iselIntExpr_R(env, e->Iex.Binop.arg2);
         addInstr(env, RISCV64Instr_ALU(RISCV64op_SUB, tmp, argL, argR));
         HReg dst = newVRegI(env);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SLTIU, dst, tmp, 1));
         return dst;
      }
      case Iop_CmpNE64:
      case Iop_CmpNE32:
      case Iop_CasCmpNE64:
      case Iop_CasCmpNE32: {
         HReg tmp  = newVRegI(env);
         HReg argL = iselIntExpr_R(env, e->Iex.Binop.arg1);
         HReg argR = iselIntExpr_R(env, e->Iex.Binop.arg2);
         addInstr(env, RISCV64Instr_ALU(RISCV64op_SUB, tmp, argL, argR));
         HReg dst = newVRegI(env);
         addInstr(env,
                  RISCV64Instr_ALU(RISCV64op_SLTU, dst, hregRISCV64_x0(), tmp));
         return dst;
      }
      case Iop_CmpLT64S:
      case Iop_CmpLT32S: {
         HReg dst  = newVRegI(env);
         HReg argL = iselIntExpr_R(env, e->Iex.Binop.arg1);
         HReg argR = iselIntExpr_R(env, e->Iex.Binop.arg2);
         addInstr(env, RISCV64Instr_ALU(RISCV64op_SLT, dst, argL, argR));
         return dst;
      }
      case Iop_CmpLE64S:
      case Iop_CmpLE32S: {
         HReg tmp  = newVRegI(env);
         HReg argL = iselIntExpr_R(env, e->Iex.Binop.arg1);
         HReg argR = iselIntExpr_R(env, e->Iex.Binop.arg2);
         addInstr(env, RISCV64Instr_ALU(RISCV64op_SLT, tmp, argR, argL));
         HReg dst = newVRegI(env);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SLTIU, dst, tmp, 1));
         return dst;
      }
      case Iop_CmpLT64U:
      case Iop_CmpLT32U: {
         HReg dst  = newVRegI(env);
         HReg argL = iselIntExpr_R(env, e->Iex.Binop.arg1);
         HReg argR = iselIntExpr_R(env, e->Iex.Binop.arg2);
         addInstr(env, RISCV64Instr_ALU(RISCV64op_SLTU, dst, argL, argR));
         return dst;
      }
      case Iop_CmpLE64U:
      case Iop_CmpLE32U: {
         HReg tmp  = newVRegI(env);
         HReg argL = iselIntExpr_R(env, e->Iex.Binop.arg1);
         HReg argR = iselIntExpr_R(env, e->Iex.Binop.arg2);
         addInstr(env, RISCV64Instr_ALU(RISCV64op_SLTU, tmp, argR, argL));
         HReg dst = newVRegI(env);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SLTIU, dst, tmp, 1));
         return dst;
      }
      case Iop_Max32U: {
         HReg argL = iselIntExpr_R(env, e->Iex.Binop.arg1);
         HReg argR = iselIntExpr_R(env, e->Iex.Binop.arg2);
         HReg cond = newVRegI(env);
         addInstr(env, RISCV64Instr_ALU(RISCV64op_SLTU, cond, argL, argR));
         HReg dst = newVRegI(env);
         addInstr(env, RISCV64Instr_CSEL(dst, argR, argL, cond));
         return dst;
      }
      case Iop_32HLto64: {
         HReg hi32s = iselIntExpr_R(env, e->Iex.Binop.arg1);
         HReg lo32s = iselIntExpr_R(env, e->Iex.Binop.arg2);

         HReg lo32_tmp = newVRegI(env);
         addInstr(env,
                  RISCV64Instr_ALUImm(RISCV64op_SLLI, lo32_tmp, lo32s, 32));
         HReg lo32 = newVRegI(env);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SRLI, lo32, lo32_tmp, 32));

         HReg hi32 = newVRegI(env);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SLLI, hi32, hi32s, 32));

         HReg dst = newVRegI(env);
         addInstr(env, RISCV64Instr_ALU(RISCV64op_OR, dst, hi32, lo32));
         return dst;
      }
      case Iop_DivModS32to32: {
         /* TODO Improve in conjunction with Iop_64HIto32. */
         HReg argL = iselIntExpr_R(env, e->Iex.Binop.arg1);
         HReg argR = iselIntExpr_R(env, e->Iex.Binop.arg2);

         HReg remw = newVRegI(env);
         addInstr(env, RISCV64Instr_ALU(RISCV64op_REMW, remw, argL, argR));
         HReg remw_hi = newVRegI(env);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SLLI, remw_hi, remw, 32));

         HReg divw = newVRegI(env);
         addInstr(env, RISCV64Instr_ALU(RISCV64op_DIVW, divw, argL, argR));
         HReg divw_hi = newVRegI(env);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SLLI, divw_hi, divw, 32));
         HReg divw_lo = newVRegI(env);
         addInstr(env,
                  RISCV64Instr_ALUImm(RISCV64op_SRLI, divw_lo, divw_hi, 32));

         HReg dst = newVRegI(env);
         addInstr(env, RISCV64Instr_ALU(RISCV64op_OR, dst, remw_hi, divw_lo));
         return dst;
      }
      case Iop_DivModU32to32: {
         /* TODO Improve in conjunction with Iop_64HIto32. */
         HReg argL = iselIntExpr_R(env, e->Iex.Binop.arg1);
         HReg argR = iselIntExpr_R(env, e->Iex.Binop.arg2);

         HReg remuw = newVRegI(env);
         addInstr(env, RISCV64Instr_ALU(RISCV64op_REMUW, remuw, argL, argR));
         HReg remuw_hi = newVRegI(env);
         addInstr(env,
                  RISCV64Instr_ALUImm(RISCV64op_SLLI, remuw_hi, remuw, 32));

         HReg divuw = newVRegI(env);
         addInstr(env, RISCV64Instr_ALU(RISCV64op_DIVUW, divuw, argL, argR));
         HReg divuw_hi = newVRegI(env);
         addInstr(env,
                  RISCV64Instr_ALUImm(RISCV64op_SLLI, divuw_hi, divuw, 32));
         HReg divuw_lo = newVRegI(env);
         addInstr(env,
                  RISCV64Instr_ALUImm(RISCV64op_SRLI, divuw_lo, divuw_hi, 32));

         HReg dst = newVRegI(env);
         addInstr(env, RISCV64Instr_ALU(RISCV64op_OR, dst, remuw_hi, divuw_lo));
         return dst;
      }
      case Iop_F32toI32S:
      case Iop_F32toI32U:
      case Iop_F32toI64S:
      case Iop_F32toI64U: {
         RISCV64FpConvertOp op;
         switch (e->Iex.Binop.op) {
         case Iop_F32toI32S:
            op = RISCV64op_FCVT_W_S;
            break;
         case Iop_F32toI32U:
            op = RISCV64op_FCVT_WU_S;
            break;
         case Iop_F32toI64S:
            op = RISCV64op_FCVT_L_S;
            break;
         case Iop_F32toI64U:
            op = RISCV64op_FCVT_LU_S;
            break;
         default:
            vassert(0);
         }
         HReg dst = newVRegI(env);
         HReg src = iselFltExpr(env, e->Iex.Binop.arg2);
         set_fcsr_rounding_mode(env, e->Iex.Binop.arg1);
         addInstr(env, RISCV64Instr_FpConvert(op, dst, src));
         return dst;
      }
      case Iop_CmpF32:
      case Iop_CmpF64: {
         HReg argL = iselFltExpr(env, e->Iex.Binop.arg1);
         HReg argR = iselFltExpr(env, e->Iex.Binop.arg2);

         HReg lt = newVRegI(env);
         HReg gt = newVRegI(env);
         HReg eq = newVRegI(env);
         if (e->Iex.Binop.op == Iop_CmpF32) {
            addInstr(env,
                     RISCV64Instr_FpCompare(RISCV64op_FLT_S, lt, argL, argR));
            addInstr(env,
                     RISCV64Instr_FpCompare(RISCV64op_FLT_S, gt, argR, argL));
            addInstr(env,
                     RISCV64Instr_FpCompare(RISCV64op_FEQ_S, eq, argL, argR));
         } else {
            addInstr(env,
                     RISCV64Instr_FpCompare(RISCV64op_FLT_D, lt, argL, argR));
            addInstr(env,
                     RISCV64Instr_FpCompare(RISCV64op_FLT_D, gt, argR, argL));
            addInstr(env,
                     RISCV64Instr_FpCompare(RISCV64op_FEQ_D, eq, argL, argR));
         }

         /*
            t0 = Ircr_UN
            t1 = Ircr_LT
            t2 = csel t1, t0, lt
            t3 = Ircr_GT
            t4 = csel t3, t2, gt
            t5 = Ircr_EQ
            dst = csel t5, t4, eq
         */
         HReg t0 = newVRegI(env);
         addInstr(env, RISCV64Instr_LI(t0, Ircr_UN));
         HReg t1 = newVRegI(env);
         addInstr(env, RISCV64Instr_LI(t1, Ircr_LT));
         HReg t2 = newVRegI(env);
         addInstr(env, RISCV64Instr_CSEL(t2, t1, t0, lt));
         HReg t3 = newVRegI(env);
         addInstr(env, RISCV64Instr_LI(t3, Ircr_GT));
         HReg t4 = newVRegI(env);
         addInstr(env, RISCV64Instr_CSEL(t4, t3, t2, gt));
         HReg t5 = newVRegI(env);
         addInstr(env, RISCV64Instr_LI(t5, Ircr_EQ));
         HReg dst = newVRegI(env);
         addInstr(env, RISCV64Instr_CSEL(dst, t5, t4, eq));
         return dst;
      }
      case Iop_F64toI32S:
      case Iop_F64toI32U:
      case Iop_F64toI64S:
      case Iop_F64toI64U: {
         RISCV64FpConvertOp op;
         switch (e->Iex.Binop.op) {
         case Iop_F64toI32S:
            op = RISCV64op_FCVT_W_D;
            break;
         case Iop_F64toI32U:
            op = RISCV64op_FCVT_WU_D;
            break;
         case Iop_F64toI64S:
            op = RISCV64op_FCVT_L_D;
            break;
         case Iop_F64toI64U:
            op = RISCV64op_FCVT_LU_D;
            break;
         default:
            vassert(0);
         }
         HReg dst = newVRegI(env);
         HReg src = iselFltExpr(env, e->Iex.Binop.arg2);
         set_fcsr_rounding_mode(env, e->Iex.Binop.arg1);
         addInstr(env, RISCV64Instr_FpConvert(op, dst, src));
         return dst;
      }
      default:
         break;
      }

      break;
   }

   /* ---------------------- UNARY OP ----------------------- */
   case Iex_Unop: {
      switch (e->Iex.Unop.op & IR_OP_MASK) {
      case Iop_Not64:
      case Iop_Not32: {
         HReg dst = newVRegI(env);
         HReg src = iselIntExpr_R(env, e->Iex.Unop.arg);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_XORI, dst, src, -1));
         return dst;
      }
      case Iop_Not1: {
         HReg dst = newVRegI(env);
         HReg src = iselIntExpr_R(env, e->Iex.Unop.arg);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SLTIU, dst, src, 1));
         return dst;
      }
      case Iop_1Uto32:
      case Iop_8Uto32:
      case Iop_8Uto64:
      case Iop_16Uto64:
      case Iop_32Uto64: {
         UInt shift = (e->Iex.Unop.op == Iop_1Uto32) ? 63 :
            64 - 8 * sizeofIRType(typeOfIRExpr(env->type_env, e->Iex.Unop.arg));
         HReg tmp = newVRegI(env);
         HReg src = iselIntExpr_R(env, e->Iex.Unop.arg);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SLLI, tmp, src, shift));
         HReg dst = newVRegI(env);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SRLI, dst, tmp, shift));
         return dst;
      }
      case Iop_1Sto8:
      case Iop_1Sto16:
      case Iop_1Sto32:
      case Iop_1Sto64: {
         HReg tmp = newVRegI(env);
         HReg src = iselIntExpr_R(env, e->Iex.Unop.arg);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SLLI, tmp, src, 63));
         HReg dst = newVRegI(env);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SRAI, dst, tmp, 63));
         return dst;
      }
      case Iop_1Uto64:
      case Iop_8Sto64:
      case Iop_16Sto64:
      case Iop_32Sto64:
         /* These are no-ops. */
         return iselIntExpr_R(env, e->Iex.Unop.arg);
      case Iop_32to1:
      case Iop_32to8:
      case Iop_32to16:
      case Iop_64to8:
      case Iop_64to16:
      case Iop_64to32: {
         UInt shift = (e->Iex.Unop.op == Iop_32to1) ? 63 :
            64 - 8 * sizeofIRType(ty);
         HReg tmp   = newVRegI(env);
         HReg src   = iselIntExpr_R(env, e->Iex.Unop.arg);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SLLI, tmp, src, shift));
         HReg dst = newVRegI(env);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SRAI, dst, tmp, shift));
         return dst;
      }
      case Iop_128HIto64: {
         HReg rHi, rLo;
         iselInt128Expr(&rHi, &rLo, env, e->Iex.Unop.arg);
         return rHi; /* and abandon rLo */
      }
      case Iop_64HIto32: {
         HReg dst = newVRegI(env);
         HReg src = iselIntExpr_R(env, e->Iex.Unop.arg);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SRAI, dst, src, 32));
         return dst;
      }
      case Iop_ReinterpF32asI32: {
         HReg dst = newVRegI(env);
         HReg src = iselFltExpr(env, e->Iex.Unop.arg);
         addInstr(env, RISCV64Instr_FpMove(RISCV64op_FMV_X_W, dst, src));
         return dst;
      }
      case Iop_ReinterpF64asI64: {
         HReg dst = newVRegI(env);
         HReg src = iselFltExpr(env, e->Iex.Unop.arg);
         addInstr(env, RISCV64Instr_FpMove(RISCV64op_FMV_X_D, dst, src));
         return dst;
      }
      case Iop_CmpNEZ8:
      case Iop_CmpNEZ16:
      case Iop_CmpNEZ32:
      case Iop_CmpNEZ64: {
         HReg dst = newVRegI(env);
         HReg src = iselIntExpr_R(env, e->Iex.Unop.arg);
         addInstr(env,
                  RISCV64Instr_ALU(RISCV64op_SLTU, dst, hregRISCV64_x0(), src));
         return dst;
      }
      case Iop_CmpwNEZ32:
      case Iop_CmpwNEZ64: {
         /* Use the fact that x | -x == 0 iff x == 0. Otherwise, either X or -X
            will have a 1 in the MSB. */
         HReg neg = newVRegI(env);
         HReg src = iselIntExpr_R(env, e->Iex.Unop.arg);
         addInstr(env,
                  RISCV64Instr_ALU(RISCV64op_SUB, neg, hregRISCV64_x0(), src));
         HReg or = newVRegI(env);
         addInstr(env, RISCV64Instr_ALU(RISCV64op_OR, or, src, neg));
         HReg dst = newVRegI(env);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SRAI, dst, or, 63));
         return dst;
      }
      case Iop_Left32:
      case Iop_Left64: {
         /* Left32/64(src) = src | -src. */
         HReg neg = newVRegI(env);
         HReg src = iselIntExpr_R(env, e->Iex.Unop.arg);
         addInstr(env,
                  RISCV64Instr_ALU(RISCV64op_SUB, neg, hregRISCV64_x0(), src));
         HReg dst = newVRegI(env);
         addInstr(env, RISCV64Instr_ALU(RISCV64op_OR, dst, src, neg));
         return dst;
      }
      case Iop_VCpop_m:
      case Iop_VFirst_m: {
         HReg dst;
         iselVecExpr_R(&dst, env, e);
         return dst;
      }

      default:
         break;
      }

      break;
   }

   /* ------------------------- GET ------------------------- */
   case Iex_Get: {
      HReg dst  = newVRegI(env);
      HReg base = get_baseblock_register();
      Int  off  = e->Iex.Get.offset - BASEBLOCK_OFFSET_ADJUSTMENT;
      vassert(off >= -2048 && off < 2048);

      if (ty == Ity_I64)
         addInstr(env, RISCV64Instr_Load(RISCV64op_LD, dst, base, off));
      else if (ty == Ity_I32)
         addInstr(env, RISCV64Instr_Load(RISCV64op_LW, dst, base, off));
      else if (ty == Ity_I16)
         addInstr(env, RISCV64Instr_Load(RISCV64op_LH, dst, base, off));
      else if (ty == Ity_I8)
         addInstr(env, RISCV64Instr_Load(RISCV64op_LB, dst, base, off));
      else
         goto irreducible;
      return dst;
   }

   /* ------------------------ CCALL ------------------------ */
   case Iex_CCall: {
      vassert(ty == e->Iex.CCall.retty);

      /* Be very restrictive for now. Only 32 and 64-bit ints are allowed for
         the return type. */
      if (e->Iex.CCall.retty != Ity_I32 && e->Iex.CCall.retty != Ity_I64)
         goto irreducible;

      /* Marshal args and do the call. */
      UInt   addToSp = 0;
      RetLoc rloc    = mk_RetLoc_INVALID();
      Bool   ok =
         doHelperCall(&addToSp, &rloc, env, NULL /*guard*/, e->Iex.CCall.cee,
                      e->Iex.CCall.retty, e->Iex.CCall.args);
      if (!ok)
         goto irreducible;
      vassert(is_sane_RetLoc(rloc));
      vassert(rloc.pri == RLPri_Int);
      vassert(addToSp == 0);

      HReg dst = newVRegI(env);
      switch (e->Iex.CCall.retty) {
      case Ity_I32:
         /* Sign-extend the value returned from the helper as is expected by the
            rest of the backend. */
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDIW, dst,
                                           hregRISCV64_x10(), 0));
         break;
      case Ity_I64:
         addInstr(env, RISCV64Instr_MV(dst, hregRISCV64_x10()));
         break;
      default:
         vassert(0);
      }
      return dst;
   }

   /* ----------------------- LITERAL ----------------------- */
   /* 64/32/16/8-bit literals. */
   case Iex_Const: {
      ULong u;
      HReg  dst = newVRegI(env);
      switch (e->Iex.Const.con->tag) {
      case Ico_U64:
         u = e->Iex.Const.con->Ico.U64;
         break;
      case Ico_U32:
         vassert(ty == Ity_I32);
         u = vex_sx_to_64(e->Iex.Const.con->Ico.U32, 32);
         break;
      case Ico_U16:
         vassert(ty == Ity_I16);
         u = vex_sx_to_64(e->Iex.Const.con->Ico.U16, 16);
         break;
      case Ico_U8:
         vassert(ty == Ity_I8);
         u = vex_sx_to_64(e->Iex.Const.con->Ico.U8, 8);
         break;
      case Ico_U1:
         vassert(ty == Ity_I1);
         u = vex_sx_to_64(e->Iex.Const.con->Ico.U1, 1);
         break;
      default:
         goto irreducible;
      }
      addInstr(env, RISCV64Instr_LI(dst, u));
      return dst;
   }

   /* ---------------------- MULTIPLEX ---------------------- */
   case Iex_ITE: {
      /* ITE(ccexpr, iftrue, iffalse) */
      if (ty == Ity_I64 || ty == Ity_I32 || ty == Ity_I16 || ty == Ity_I8) {
         HReg dst     = newVRegI(env);
         HReg iftrue  = iselIntExpr_R(env, e->Iex.ITE.iftrue);
         HReg iffalse = iselIntExpr_R(env, e->Iex.ITE.iffalse);
         HReg cond    = iselIntExpr_R(env, e->Iex.ITE.cond);
         addInstr(env, RISCV64Instr_CSEL(dst, iftrue, iffalse, cond));
         return dst;
      }
      break;
   }

   default:
      break;
   }

   /* We get here if no pattern matched. */
irreducible:
   ppIRExpr(e);
   vpanic("iselIntExpr_R(riscv64)");
}

static HReg iselIntExpr_R(ISelEnv* env, IRExpr* e)
{
   HReg r = iselIntExpr_R_wrk(env, e);

   /* Sanity checks ... */
   vassert(hregClass(r) == HRcInt64);
   vassert(hregIsVirtual(r));

   return r;
}

/*------------------------------------------------------------*/
/*--- ISEL: Integer expressions (128 bit)                  ---*/
/*------------------------------------------------------------*/

/* DO NOT CALL THIS DIRECTLY ! */
static void iselInt128Expr_wrk(HReg* rHi, HReg* rLo, ISelEnv* env, IRExpr* e)
{
   vassert(typeOfIRExpr(env->type_env, e) == Ity_I128);

   /* ---------------------- BINARY OP ---------------------- */
   if (e->tag == Iex_Binop) {
      switch (e->Iex.Binop.op) {
      /* 64 x 64 -> 128 multiply */
      case Iop_MullS64:
      case Iop_MullU64: {
         HReg argL = iselIntExpr_R(env, e->Iex.Binop.arg1);
         HReg argR = iselIntExpr_R(env, e->Iex.Binop.arg2);
         *rHi      = newVRegI(env);
         *rLo      = newVRegI(env);
         if (e->Iex.Binop.op == Iop_MullS64)
            addInstr(env, RISCV64Instr_ALU(RISCV64op_MULH, *rHi, argL, argR));
         else
            addInstr(env, RISCV64Instr_ALU(RISCV64op_MULHU, *rHi, argL, argR));
         addInstr(env, RISCV64Instr_ALU(RISCV64op_MUL, *rLo, argL, argR));
         return;
      }

      /* 64 x 64 -> (64(rem),64(div)) division */
      case Iop_DivModS64to64:
      case Iop_DivModU64to64: {
         HReg argL = iselIntExpr_R(env, e->Iex.Binop.arg1);
         HReg argR = iselIntExpr_R(env, e->Iex.Binop.arg2);
         *rHi      = newVRegI(env);
         *rLo      = newVRegI(env);
         if (e->Iex.Binop.op == Iop_DivModS64to64) {
            addInstr(env, RISCV64Instr_ALU(RISCV64op_REM, *rHi, argL, argR));
            addInstr(env, RISCV64Instr_ALU(RISCV64op_DIV, *rLo, argL, argR));
         } else {
            addInstr(env, RISCV64Instr_ALU(RISCV64op_REMU, *rHi, argL, argR));
            addInstr(env, RISCV64Instr_ALU(RISCV64op_DIVU, *rLo, argL, argR));
         }
         return;
      }

      /* 64HLto128(e1,e2) */
      case Iop_64HLto128:
         *rHi = iselIntExpr_R(env, e->Iex.Binop.arg1);
         *rLo = iselIntExpr_R(env, e->Iex.Binop.arg2);
         return;

      default:
         break;
      }
   }

   ppIRExpr(e);
   vpanic("iselInt128Expr(riscv64)");
}

/* Compute a 128-bit value into a register pair, which is returned as the first
   two parameters. As with iselIntExpr_R, these will be virtual registers and
   they must not be changed by subsequent code emitted by the caller. */
static void iselInt128Expr(HReg* rHi, HReg* rLo, ISelEnv* env, IRExpr* e)
{
   iselInt128Expr_wrk(rHi, rLo, env, e);

   /* Sanity checks ... */
   vassert(hregClass(*rHi) == HRcInt64);
   vassert(hregIsVirtual(*rHi));
   vassert(hregClass(*rLo) == HRcInt64);
   vassert(hregIsVirtual(*rLo));
}

#define RVVCALL(macro, ...)  macro(__VA_ARGS__)

typedef Char int8_t;
typedef Short int16_t;
typedef Int int32_t;
typedef Long int64_t;
typedef UChar uint8_t;
typedef UShort uint16_t;
typedef UInt uint32_t;
typedef ULong uint64_t;

/* (TD, T2, TX2) */
#define OP_SS_B int8_t, int8_t, int8_t
#define OP_SS_H int16_t, int16_t, int16_t
#define OP_SS_W int32_t, int32_t, int32_t
#define OP_SS_D int64_t, int64_t, int64_t

/* (TD, T1, T2, TX1, TX2) */
#define OP_SSS_B int8_t, int8_t, int8_t, int8_t, int8_t
#define OP_SSS_H int16_t, int16_t, int16_t, int16_t, int16_t
#define OP_SSS_W int32_t, int32_t, int32_t, int32_t, int32_t
#define OP_SSS_D int64_t, int64_t, int64_t, int64_t, int64_t

#define OP_UUU_B uint8_t, uint8_t, uint8_t, uint8_t, uint8_t
#define OP_UUU_H uint16_t, uint16_t, uint16_t, uint16_t, uint16_t
#define OP_UUU_W uint32_t, uint32_t, uint32_t, uint32_t, uint32_t
#define OP_UUU_D uint64_t, uint64_t, uint64_t, uint64_t, uint64_t

#define OP_SUS_B int8_t, uint8_t, int8_t, uint8_t, int8_t
#define OP_SUS_H int16_t, uint16_t, int16_t, uint16_t, int16_t
#define OP_SUS_W int32_t, uint32_t, int32_t, uint32_t, int32_t
#define OP_SUS_D int64_t, uint64_t, int64_t, uint64_t, int64_t

#define DO_AND(N, M)  (N & M)
#define DO_OR(N, M)   (N | M)
#define DO_XOR(N, M)  (N ^ M)
#define DO_ADD(N, M)  (N + M)
#define DO_SUB(N, M)  (N - M)
#define DO_RSUB(N, M) (M - N)

/* Signed min/max */
#define DO_MAX(N, M)  ((N) >= (M) ? (N) : (M))
#define DO_MIN(N, M)  ((N) >= (M) ? (M) : (N))

#define DO_MUL(N, M)  (N * M)

#ifndef unlikely
#define unlikely(x) (x)
#endif

/* Vector Integer Divide Instructions */
#define DO_DIVU(N, M) (unlikely(M == 0) ? (__typeof(N))(-1) : N / M)
#define DO_REMU(N, M) (unlikely(M == 0) ? N : N % M)
#define DO_DIV(N, M)  (unlikely(M == 0) ? (__typeof(N))(-1) : \
        unlikely((N == -N) && (M == (__typeof(N))(-1))) ? N : N / M)
#define DO_REM(N, M)  (unlikely(M == 0) ? N : \
        unlikely((N == -N) && (M == (__typeof(N))(-1))) ? 0 : N % M)

/** BEGIN: high bits of multiply **/

/* Many of helpers for vector are copied from QEMU */

/* Long integer helpers */
static inline void mul64(uint64_t *plow, uint64_t *phigh,
                         uint64_t a, uint64_t b)
{
    typedef union {
        uint64_t ll;
        struct {
#if HOST_BIG_ENDIAN
            uint32_t high, low;
#else
            uint32_t low, high;
#endif
        } l;
    } LL;
    LL rl, rm, rn, rh, a0, b0;
    uint64_t c;

    a0.ll = a;
    b0.ll = b;

    rl.ll = (uint64_t)a0.l.low * b0.l.low;
    rm.ll = (uint64_t)a0.l.low * b0.l.high;
    rn.ll = (uint64_t)a0.l.high * b0.l.low;
    rh.ll = (uint64_t)a0.l.high * b0.l.high;

    c = (uint64_t)rl.l.high + rm.l.low + rn.l.low;
    rl.l.high = c;
    c >>= 32;
    c = c + rm.l.high + rn.l.high + rh.l.low;
    rh.l.low = c;
    rh.l.high += (uint32_t)(c >> 32);

    *plow = rl.ll;
    *phigh = rh.ll;
}

/* Unsigned 64x64 -> 128 multiplication */
static void mulu64 (uint64_t *plow, uint64_t *phigh, uint64_t a, uint64_t b)
{
    mul64(plow, phigh, a, b);
}

/* Signed 64x64 -> 128 multiplication */
static void muls64 (uint64_t *plow, uint64_t *phigh, int64_t a, int64_t b)
{
    uint64_t rh;

    mul64(plow, &rh, a, b);

    /* Adjust for signs.  */
    if (b < 0) {
        rh -= a;
    }
    if (a < 0) {
        rh -= b;
    }
    *phigh = rh;
}

static int8_t do_mulh_b(int8_t s2, int8_t s1)
{
    return (int16_t)s2 * (int16_t)s1 >> 8;
}

static int16_t do_mulh_h(int16_t s2, int16_t s1)
{
    return (int32_t)s2 * (int32_t)s1 >> 16;
}

static int32_t do_mulh_w(int32_t s2, int32_t s1)
{
    return (int64_t)s2 * (int64_t)s1 >> 32;
}

static int64_t do_mulh_d(int64_t s2, int64_t s1)
{
    uint64_t hi_64, lo_64;

    muls64(&lo_64, &hi_64, s1, s2);
    return hi_64;
}

static uint8_t do_mulhu_b(uint8_t s2, uint8_t s1)
{
    return (uint16_t)s2 * (uint16_t)s1 >> 8;
}

static uint16_t do_mulhu_h(uint16_t s2, uint16_t s1)
{
    return (uint32_t)s2 * (uint32_t)s1 >> 16;
}

static uint32_t do_mulhu_w(uint32_t s2, uint32_t s1)
{
    return (uint64_t)s2 * (uint64_t)s1 >> 32;
}

static uint64_t do_mulhu_d(uint64_t s2, uint64_t s1)
{
    uint64_t hi_64, lo_64;

    mulu64(&lo_64, &hi_64, s2, s1);
    return hi_64;
}

static int8_t do_mulhsu_b(int8_t s2, uint8_t s1)
{
    return (int16_t)s2 * (uint16_t)s1 >> 8;
}

static int16_t do_mulhsu_h(int16_t s2, uint16_t s1)
{
    return (int32_t)s2 * (uint32_t)s1 >> 16;
}

static int32_t do_mulhsu_w(int32_t s2, uint32_t s1)
{
    return (int64_t)s2 * (uint64_t)s1 >> 32;
}

/*
 * Let  A = signed operand,
 *      B = unsigned operand
 *      P = mulu64(A, B), unsigned product
 *
 * LET  X = 2 ** 64  - A, 2's complement of A
 *      SP = signed product
 * THEN
 *      IF A < 0
 *          SP = -X * B
 *             = -(2 ** 64 - A) * B
 *             = A * B - 2 ** 64 * B
 *             = P - 2 ** 64 * B
 *      ELSE
 *          SP = P
 * THEN
 *      HI_P -= (A < 0 ? B : 0)
 */

static int64_t do_mulhsu_d(int64_t s2, uint64_t s1)
{
    uint64_t hi_64, lo_64;

    mulu64(&lo_64, &hi_64, s2, s1);

    hi_64 -= s2 < 0 ? s1 : 0;
    return hi_64;
}

/** END: high bits of multiply **/

#define DO_NOT(N)    (~N)
#define DO_CMPNEZ(N) ((N != 0) ? -1LL : 0)

#define OPIVV1(NAME, TD, T2, TX2, OP)                  \
static void do_##NAME(void *vd, void *vs2, int i)      \
{                                                      \
    TX2 s2 = *((T2 *)vs2 + i);                         \
    *((TD *)vd + i) = OP(s2);                          \
}

RVVCALL(OPIVV1, VNot_8, OP_SS_B, DO_NOT)
RVVCALL(OPIVV1, VNot_16, OP_SS_H, DO_NOT)
RVVCALL(OPIVV1, VNot_32, OP_SS_W, DO_NOT)
RVVCALL(OPIVV1, VNot_64, OP_SS_D, DO_NOT)

RVVCALL(OPIVV1, VCmpNEZ_8, OP_SS_B, DO_CMPNEZ)
RVVCALL(OPIVV1, VCmpNEZ_16, OP_SS_H, DO_CMPNEZ)
RVVCALL(OPIVV1, VCmpNEZ_32, OP_SS_W, DO_CMPNEZ)
RVVCALL(OPIVV1, VCmpNEZ_64, OP_SS_D, DO_CMPNEZ)

#define GEN_VEXT_V(NAME)                               \
static void h_Iop_##NAME(void *vd, void *vs2, int len) \
{                                                      \
   for (int i = 0; i < len; ++i) {                     \
      do_##NAME(vd, vs2, i);                           \
   }                                                   \
}

GEN_VEXT_V(VNot_8)
GEN_VEXT_V(VNot_16)
GEN_VEXT_V(VNot_32)
GEN_VEXT_V(VNot_64)

GEN_VEXT_V(VCmpNEZ_8)
GEN_VEXT_V(VCmpNEZ_16)
GEN_VEXT_V(VCmpNEZ_32)
GEN_VEXT_V(VCmpNEZ_64)

#define GEN_VEXT_V_EXPANDBIT(TD, BITS)                                  \
static void h_Iop_VExpandBitsTo_##BITS(void *dst, void *src, int len)   \
{                                                                       \
   TD *d = dst;                                                         \
   char *s = src;                                                       \
                                                                        \
   for (int i = 0; i < len; i += 8) {                                   \
      for (int j = i, bit = 0; j < len && bit < 8; ++j, ++bit) {        \
         d[j] = (*s & (1 << bit)) ? -1LL : 0;                           \
      }                                                                 \
      ++s;                                                              \
   }                                                                    \
}

GEN_VEXT_V_EXPANDBIT(int8_t, 8)
GEN_VEXT_V_EXPANDBIT(int16_t, 16)
GEN_VEXT_V_EXPANDBIT(int32_t, 32)
GEN_VEXT_V_EXPANDBIT(int64_t, 64)

#define OPIVV2(NAME, TD, T1, T2, TX1, TX2, OP)                \
static void do_##NAME(void *vd, void *vs1, void *vs2, int i)  \
{                                                             \
   TX1 s1 = *((T1 *)vs1 + i);                                 \
   TX2 s2 = *((T2 *)vs2 + i);                                 \
   *((TD *)vd + i) = OP(s2, s1);                              \
}

RVVCALL(OPIVV2, VAnd_vv_8, OP_SSS_B, DO_AND)
RVVCALL(OPIVV2, VAnd_vv_16, OP_SSS_H, DO_AND)
RVVCALL(OPIVV2, VAnd_vv_32, OP_SSS_W, DO_AND)
RVVCALL(OPIVV2, VAnd_vv_64, OP_SSS_D, DO_AND)

RVVCALL(OPIVV2, VOr_vv_8, OP_SSS_B, DO_OR)
RVVCALL(OPIVV2, VOr_vv_16, OP_SSS_H, DO_OR)
RVVCALL(OPIVV2, VOr_vv_32, OP_SSS_W, DO_OR)
RVVCALL(OPIVV2, VOr_vv_64, OP_SSS_D, DO_OR)

RVVCALL(OPIVV2, VXor_vv_8, OP_SSS_B, DO_XOR)
RVVCALL(OPIVV2, VXor_vv_16, OP_SSS_H, DO_XOR)
RVVCALL(OPIVV2, VXor_vv_32, OP_SSS_W, DO_XOR)
RVVCALL(OPIVV2, VXor_vv_64, OP_SSS_D, DO_XOR)

RVVCALL(OPIVV2, VAdd_vv_8, OP_SSS_B, DO_ADD)
RVVCALL(OPIVV2, VAdd_vv_16, OP_SSS_H, DO_ADD)
RVVCALL(OPIVV2, VAdd_vv_32, OP_SSS_W, DO_ADD)
RVVCALL(OPIVV2, VAdd_vv_64, OP_SSS_D, DO_ADD)

RVVCALL(OPIVV2, VSub_vv_8, OP_SSS_B, DO_SUB)
RVVCALL(OPIVV2, VSub_vv_16, OP_SSS_H, DO_SUB)
RVVCALL(OPIVV2, VSub_vv_32, OP_SSS_W, DO_SUB)
RVVCALL(OPIVV2, VSub_vv_64, OP_SSS_D, DO_SUB)

RVVCALL(OPIVV2, VMinu_vv_8, OP_UUU_B, DO_MIN)
RVVCALL(OPIVV2, VMinu_vv_16, OP_UUU_H, DO_MIN)
RVVCALL(OPIVV2, VMinu_vv_32, OP_UUU_W, DO_MIN)
RVVCALL(OPIVV2, VMinu_vv_64, OP_UUU_D, DO_MIN)
RVVCALL(OPIVV2, VMin_vv_8, OP_SSS_B, DO_MIN)
RVVCALL(OPIVV2, VMin_vv_16, OP_SSS_H, DO_MIN)
RVVCALL(OPIVV2, VMin_vv_32, OP_SSS_W, DO_MIN)
RVVCALL(OPIVV2, VMin_vv_64, OP_SSS_D, DO_MIN)

RVVCALL(OPIVV2, VMaxu_vv_8, OP_UUU_B, DO_MAX)
RVVCALL(OPIVV2, VMaxu_vv_16, OP_UUU_H, DO_MAX)
RVVCALL(OPIVV2, VMaxu_vv_32, OP_UUU_W, DO_MAX)
RVVCALL(OPIVV2, VMaxu_vv_64, OP_UUU_D, DO_MAX)
RVVCALL(OPIVV2, VMax_vv_8, OP_SSS_B, DO_MAX)
RVVCALL(OPIVV2, VMax_vv_16, OP_SSS_H, DO_MAX)
RVVCALL(OPIVV2, VMax_vv_32, OP_SSS_W, DO_MAX)
RVVCALL(OPIVV2, VMax_vv_64, OP_SSS_D, DO_MAX)

RVVCALL(OPIVV2, VMul_vv_8, OP_SSS_B, DO_MUL)
RVVCALL(OPIVV2, VMul_vv_16, OP_SSS_H, DO_MUL)
RVVCALL(OPIVV2, VMul_vv_32, OP_SSS_W, DO_MUL)
RVVCALL(OPIVV2, VMul_vv_64, OP_SSS_D, DO_MUL)
RVVCALL(OPIVV2, VMulh_vv_8, OP_SSS_B, do_mulh_b)
RVVCALL(OPIVV2, VMulh_vv_16, OP_SSS_H, do_mulh_h)
RVVCALL(OPIVV2, VMulh_vv_32, OP_SSS_W, do_mulh_w)
RVVCALL(OPIVV2, VMulh_vv_64, OP_SSS_D, do_mulh_d)
RVVCALL(OPIVV2, VMulhu_vv_8, OP_UUU_B, do_mulhu_b)
RVVCALL(OPIVV2, VMulhu_vv_16, OP_UUU_H, do_mulhu_h)
RVVCALL(OPIVV2, VMulhu_vv_32, OP_UUU_W, do_mulhu_w)
RVVCALL(OPIVV2, VMulhu_vv_64, OP_UUU_D, do_mulhu_d)
RVVCALL(OPIVV2, VMulhsu_vv_8, OP_SUS_B, do_mulhsu_b)
RVVCALL(OPIVV2, VMulhsu_vv_16, OP_SUS_H, do_mulhsu_h)
RVVCALL(OPIVV2, VMulhsu_vv_32, OP_SUS_W, do_mulhsu_w)
RVVCALL(OPIVV2, VMulhsu_vv_64, OP_SUS_D, do_mulhsu_d)

RVVCALL(OPIVV2, VDivu_vv_8, OP_UUU_B, DO_DIVU)
RVVCALL(OPIVV2, VDivu_vv_16, OP_UUU_H, DO_DIVU)
RVVCALL(OPIVV2, VDivu_vv_32, OP_UUU_W, DO_DIVU)
RVVCALL(OPIVV2, VDivu_vv_64, OP_UUU_D, DO_DIVU)
RVVCALL(OPIVV2, VDiv_vv_8, OP_SSS_B, DO_DIV)
RVVCALL(OPIVV2, VDiv_vv_16, OP_SSS_H, DO_DIV)
RVVCALL(OPIVV2, VDiv_vv_32, OP_SSS_W, DO_DIV)
RVVCALL(OPIVV2, VDiv_vv_64, OP_SSS_D, DO_DIV)
RVVCALL(OPIVV2, VRemu_vv_8, OP_UUU_B, DO_REMU)
RVVCALL(OPIVV2, VRemu_vv_16, OP_UUU_H, DO_REMU)
RVVCALL(OPIVV2, VRemu_vv_32, OP_UUU_W, DO_REMU)
RVVCALL(OPIVV2, VRemu_vv_64, OP_UUU_D, DO_REMU)
RVVCALL(OPIVV2, VRem_vv_8, OP_SSS_B, DO_REM)
RVVCALL(OPIVV2, VRem_vv_16, OP_SSS_H, DO_REM)
RVVCALL(OPIVV2, VRem_vv_32, OP_SSS_W, DO_REM)
RVVCALL(OPIVV2, VRem_vv_64, OP_SSS_D, DO_REM)

typedef void opivv2_fn(void *vd, void *vs1, void *vs2, int i);

static void do_vext_vv(void *vd, void *vs1, void *vs2, opivv2_fn *fn, int len)
{
   for (int i = 0; i < len; ++i) {
      fn(vd, vs1, vs2, i);
   }
}

#define GEN_VEXT_VV(NAME)                                          \
static void h_Iop_##NAME(void *vd, void *vs1, void *vs2, int len)  \
{                                                                  \
    do_vext_vv(vd, vs1, vs2, do_##NAME, len);                      \
}

GEN_VEXT_VV(VAnd_vv_8)
GEN_VEXT_VV(VAnd_vv_16)
GEN_VEXT_VV(VAnd_vv_32)
GEN_VEXT_VV(VAnd_vv_64)

GEN_VEXT_VV(VOr_vv_8)
GEN_VEXT_VV(VOr_vv_16)
GEN_VEXT_VV(VOr_vv_32)
GEN_VEXT_VV(VOr_vv_64)

GEN_VEXT_VV(VXor_vv_8)
GEN_VEXT_VV(VXor_vv_16)
GEN_VEXT_VV(VXor_vv_32)
GEN_VEXT_VV(VXor_vv_64)

GEN_VEXT_VV(VAdd_vv_8)
GEN_VEXT_VV(VAdd_vv_16)
GEN_VEXT_VV(VAdd_vv_32)
GEN_VEXT_VV(VAdd_vv_64)

GEN_VEXT_VV(VSub_vv_8)
GEN_VEXT_VV(VSub_vv_16)
GEN_VEXT_VV(VSub_vv_32)
GEN_VEXT_VV(VSub_vv_64)

GEN_VEXT_VV(VMinu_vv_8)
GEN_VEXT_VV(VMinu_vv_16)
GEN_VEXT_VV(VMinu_vv_32)
GEN_VEXT_VV(VMinu_vv_64)
GEN_VEXT_VV(VMin_vv_8)
GEN_VEXT_VV(VMin_vv_16)
GEN_VEXT_VV(VMin_vv_32)
GEN_VEXT_VV(VMin_vv_64)

GEN_VEXT_VV(VMaxu_vv_8)
GEN_VEXT_VV(VMaxu_vv_16)
GEN_VEXT_VV(VMaxu_vv_32)
GEN_VEXT_VV(VMaxu_vv_64)
GEN_VEXT_VV(VMax_vv_8)
GEN_VEXT_VV(VMax_vv_16)
GEN_VEXT_VV(VMax_vv_32)
GEN_VEXT_VV(VMax_vv_64)

GEN_VEXT_VV(VMul_vv_8)
GEN_VEXT_VV(VMul_vv_16)
GEN_VEXT_VV(VMul_vv_32)
GEN_VEXT_VV(VMul_vv_64)
GEN_VEXT_VV(VMulh_vv_8)
GEN_VEXT_VV(VMulh_vv_16)
GEN_VEXT_VV(VMulh_vv_32)
GEN_VEXT_VV(VMulh_vv_64)
GEN_VEXT_VV(VMulhu_vv_8)
GEN_VEXT_VV(VMulhu_vv_16)
GEN_VEXT_VV(VMulhu_vv_32)
GEN_VEXT_VV(VMulhu_vv_64)
GEN_VEXT_VV(VMulhsu_vv_8)
GEN_VEXT_VV(VMulhsu_vv_16)
GEN_VEXT_VV(VMulhsu_vv_32)
GEN_VEXT_VV(VMulhsu_vv_64)

GEN_VEXT_VV(VDivu_vv_8)
GEN_VEXT_VV(VDivu_vv_16)
GEN_VEXT_VV(VDivu_vv_32)
GEN_VEXT_VV(VDivu_vv_64)
GEN_VEXT_VV(VDiv_vv_8)
GEN_VEXT_VV(VDiv_vv_16)
GEN_VEXT_VV(VDiv_vv_32)
GEN_VEXT_VV(VDiv_vv_64)
GEN_VEXT_VV(VRemu_vv_8)
GEN_VEXT_VV(VRemu_vv_16)
GEN_VEXT_VV(VRemu_vv_32)
GEN_VEXT_VV(VRemu_vv_64)
GEN_VEXT_VV(VRem_vv_8)
GEN_VEXT_VV(VRem_vv_16)
GEN_VEXT_VV(VRem_vv_32)
GEN_VEXT_VV(VRem_vv_64)

/*
 * (T1)s1 gives the real operator type.
 * (TX1)(T1)s1 expands the operator type of widen or narrow operations.
 */
#define OPIVX2(NAME, TD, T1, T2, TX1, TX2, OP)             \
static void do_##NAME(void *vd, Long s1, void *vs2, int i) \
{                                                          \
    TX2 s2 = *((T2 *)vs2 + i);                             \
    *((TD *)vd + i) = OP(s2, (TX1)(T1)s1);                 \
}

RVVCALL(OPIVX2, VAnd_vx_8, OP_SSS_B, DO_AND)
RVVCALL(OPIVX2, VAnd_vx_16, OP_SSS_H, DO_AND)
RVVCALL(OPIVX2, VAnd_vx_32, OP_SSS_W, DO_AND)
RVVCALL(OPIVX2, VAnd_vx_64, OP_SSS_D, DO_AND)

RVVCALL(OPIVX2, VOr_vx_8, OP_SSS_B, DO_OR)
RVVCALL(OPIVX2, VOr_vx_16, OP_SSS_H, DO_OR)
RVVCALL(OPIVX2, VOr_vx_32, OP_SSS_W, DO_OR)
RVVCALL(OPIVX2, VOr_vx_64, OP_SSS_D, DO_OR)

RVVCALL(OPIVX2, VXor_vx_8, OP_SSS_B, DO_XOR)
RVVCALL(OPIVX2, VXor_vx_16, OP_SSS_H, DO_XOR)
RVVCALL(OPIVX2, VXor_vx_32, OP_SSS_W, DO_XOR)
RVVCALL(OPIVX2, VXor_vx_64, OP_SSS_D, DO_XOR)

RVVCALL(OPIVX2, VAdd_vx_8, OP_SSS_B, DO_ADD)
RVVCALL(OPIVX2, VAdd_vx_16, OP_SSS_H, DO_ADD)
RVVCALL(OPIVX2, VAdd_vx_32, OP_SSS_W, DO_ADD)
RVVCALL(OPIVX2, VAdd_vx_64, OP_SSS_D, DO_ADD)

RVVCALL(OPIVX2, VSub_vx_8, OP_SSS_B, DO_SUB)
RVVCALL(OPIVX2, VSub_vx_16, OP_SSS_H, DO_SUB)
RVVCALL(OPIVX2, VSub_vx_32, OP_SSS_W, DO_SUB)
RVVCALL(OPIVX2, VSub_vx_64, OP_SSS_D, DO_SUB)

RVVCALL(OPIVX2, VRsub_vx_8, OP_SSS_B, DO_RSUB)
RVVCALL(OPIVX2, VRsub_vx_16, OP_SSS_H, DO_RSUB)
RVVCALL(OPIVX2, VRsub_vx_32, OP_SSS_W, DO_RSUB)
RVVCALL(OPIVX2, VRsub_vx_64, OP_SSS_D, DO_RSUB)

RVVCALL(OPIVX2, VMinu_vx_8, OP_UUU_B, DO_MIN)
RVVCALL(OPIVX2, VMinu_vx_16, OP_UUU_H, DO_MIN)
RVVCALL(OPIVX2, VMinu_vx_32, OP_UUU_W, DO_MIN)
RVVCALL(OPIVX2, VMinu_vx_64, OP_UUU_D, DO_MIN)
RVVCALL(OPIVX2, VMin_vx_8, OP_SSS_B, DO_MIN)
RVVCALL(OPIVX2, VMin_vx_16, OP_SSS_H, DO_MIN)
RVVCALL(OPIVX2, VMin_vx_32, OP_SSS_W, DO_MIN)
RVVCALL(OPIVX2, VMin_vx_64, OP_SSS_D, DO_MIN)

RVVCALL(OPIVX2, VMaxu_vx_8, OP_UUU_B, DO_MAX)
RVVCALL(OPIVX2, VMaxu_vx_16, OP_UUU_H, DO_MAX)
RVVCALL(OPIVX2, VMaxu_vx_32, OP_UUU_W, DO_MAX)
RVVCALL(OPIVX2, VMaxu_vx_64, OP_UUU_D, DO_MAX)
RVVCALL(OPIVX2, VMax_vx_8, OP_SSS_B, DO_MAX)
RVVCALL(OPIVX2, VMax_vx_16, OP_SSS_H, DO_MAX)
RVVCALL(OPIVX2, VMax_vx_32, OP_SSS_W, DO_MAX)
RVVCALL(OPIVX2, VMax_vx_64, OP_SSS_D, DO_MAX)

RVVCALL(OPIVX2, VMul_vx_8, OP_SSS_B, DO_MUL)
RVVCALL(OPIVX2, VMul_vx_16, OP_SSS_H, DO_MUL)
RVVCALL(OPIVX2, VMul_vx_32, OP_SSS_W, DO_MUL)
RVVCALL(OPIVX2, VMul_vx_64, OP_SSS_D, DO_MUL)
RVVCALL(OPIVX2, VMulh_vx_8, OP_SSS_B, do_mulh_b)
RVVCALL(OPIVX2, VMulh_vx_16, OP_SSS_H, do_mulh_h)
RVVCALL(OPIVX2, VMulh_vx_32, OP_SSS_W, do_mulh_w)
RVVCALL(OPIVX2, VMulh_vx_64, OP_SSS_D, do_mulh_d)
RVVCALL(OPIVX2, VMulhu_vx_8, OP_UUU_B, do_mulhu_b)
RVVCALL(OPIVX2, VMulhu_vx_16, OP_UUU_H, do_mulhu_h)
RVVCALL(OPIVX2, VMulhu_vx_32, OP_UUU_W, do_mulhu_w)
RVVCALL(OPIVX2, VMulhu_vx_64, OP_UUU_D, do_mulhu_d)
RVVCALL(OPIVX2, VMulhsu_vx_8, OP_SUS_B, do_mulhsu_b)
RVVCALL(OPIVX2, VMulhsu_vx_16, OP_SUS_H, do_mulhsu_h)
RVVCALL(OPIVX2, VMulhsu_vx_32, OP_SUS_W, do_mulhsu_w)
RVVCALL(OPIVX2, VMulhsu_vx_64, OP_SUS_D, do_mulhsu_d)

RVVCALL(OPIVX2, VDivu_vx_8, OP_UUU_B, DO_DIVU)
RVVCALL(OPIVX2, VDivu_vx_16, OP_UUU_H, DO_DIVU)
RVVCALL(OPIVX2, VDivu_vx_32, OP_UUU_W, DO_DIVU)
RVVCALL(OPIVX2, VDivu_vx_64, OP_UUU_D, DO_DIVU)
RVVCALL(OPIVX2, VDiv_vx_8, OP_SSS_B, DO_DIV)
RVVCALL(OPIVX2, VDiv_vx_16, OP_SSS_H, DO_DIV)
RVVCALL(OPIVX2, VDiv_vx_32, OP_SSS_W, DO_DIV)
RVVCALL(OPIVX2, VDiv_vx_64, OP_SSS_D, DO_DIV)
RVVCALL(OPIVX2, VRemu_vx_8, OP_UUU_B, DO_REMU)
RVVCALL(OPIVX2, VRemu_vx_16, OP_UUU_H, DO_REMU)
RVVCALL(OPIVX2, VRemu_vx_32, OP_UUU_W, DO_REMU)
RVVCALL(OPIVX2, VRemu_vx_64, OP_UUU_D, DO_REMU)
RVVCALL(OPIVX2, VRem_vx_8, OP_SSS_B, DO_REM)
RVVCALL(OPIVX2, VRem_vx_16, OP_SSS_H, DO_REM)
RVVCALL(OPIVX2, VRem_vx_32, OP_SSS_W, DO_REM)
RVVCALL(OPIVX2, VRem_vx_64, OP_SSS_D, DO_REM)

typedef void opivx2_fn(void *vd, Long s1, void *vs2, int i);

static void do_vext_vx(void *vd, Long s1, void *vs2, opivx2_fn *fn, int len)
{
   for (int i = 0; i < len; ++i) {
      fn(vd, s1, vs2, i);
   }
}

#define GEN_VEXT_VX(NAME)                                        \
static void h_Iop_##NAME(void *vd, Long s1, void *vs2, int len)  \
{                                                                \
    do_vext_vx(vd, s1, vs2, do_##NAME, len);                     \
}

GEN_VEXT_VX(VAnd_vx_8)
GEN_VEXT_VX(VAnd_vx_16)
GEN_VEXT_VX(VAnd_vx_32)
GEN_VEXT_VX(VAnd_vx_64)

GEN_VEXT_VX(VOr_vx_8)
GEN_VEXT_VX(VOr_vx_16)
GEN_VEXT_VX(VOr_vx_32)
GEN_VEXT_VX(VOr_vx_64)

GEN_VEXT_VX(VXor_vx_8)
GEN_VEXT_VX(VXor_vx_16)
GEN_VEXT_VX(VXor_vx_32)
GEN_VEXT_VX(VXor_vx_64)

GEN_VEXT_VX(VAdd_vx_8)
GEN_VEXT_VX(VAdd_vx_16)
GEN_VEXT_VX(VAdd_vx_32)
GEN_VEXT_VX(VAdd_vx_64)

GEN_VEXT_VX(VSub_vx_8)
GEN_VEXT_VX(VSub_vx_16)
GEN_VEXT_VX(VSub_vx_32)
GEN_VEXT_VX(VSub_vx_64)

GEN_VEXT_VX(VRsub_vx_8)
GEN_VEXT_VX(VRsub_vx_16)
GEN_VEXT_VX(VRsub_vx_32)
GEN_VEXT_VX(VRsub_vx_64)

GEN_VEXT_VX(VMinu_vx_8)
GEN_VEXT_VX(VMinu_vx_16)
GEN_VEXT_VX(VMinu_vx_32)
GEN_VEXT_VX(VMinu_vx_64)
GEN_VEXT_VX(VMin_vx_8)
GEN_VEXT_VX(VMin_vx_16)
GEN_VEXT_VX(VMin_vx_32)
GEN_VEXT_VX(VMin_vx_64)

GEN_VEXT_VX(VMaxu_vx_8)
GEN_VEXT_VX(VMaxu_vx_16)
GEN_VEXT_VX(VMaxu_vx_32)
GEN_VEXT_VX(VMaxu_vx_64)
GEN_VEXT_VX(VMax_vx_8)
GEN_VEXT_VX(VMax_vx_16)
GEN_VEXT_VX(VMax_vx_32)
GEN_VEXT_VX(VMax_vx_64)

GEN_VEXT_VX(VMul_vx_8)
GEN_VEXT_VX(VMul_vx_16)
GEN_VEXT_VX(VMul_vx_32)
GEN_VEXT_VX(VMul_vx_64)
GEN_VEXT_VX(VMulh_vx_8)
GEN_VEXT_VX(VMulh_vx_16)
GEN_VEXT_VX(VMulh_vx_32)
GEN_VEXT_VX(VMulh_vx_64)
GEN_VEXT_VX(VMulhu_vx_8)
GEN_VEXT_VX(VMulhu_vx_16)
GEN_VEXT_VX(VMulhu_vx_32)
GEN_VEXT_VX(VMulhu_vx_64)
GEN_VEXT_VX(VMulhsu_vx_8)
GEN_VEXT_VX(VMulhsu_vx_16)
GEN_VEXT_VX(VMulhsu_vx_32)
GEN_VEXT_VX(VMulhsu_vx_64)

GEN_VEXT_VX(VDivu_vx_8)
GEN_VEXT_VX(VDivu_vx_16)
GEN_VEXT_VX(VDivu_vx_32)
GEN_VEXT_VX(VDivu_vx_64)
GEN_VEXT_VX(VDiv_vx_8)
GEN_VEXT_VX(VDiv_vx_16)
GEN_VEXT_VX(VDiv_vx_32)
GEN_VEXT_VX(VDiv_vx_64)
GEN_VEXT_VX(VRemu_vx_8)
GEN_VEXT_VX(VRemu_vx_16)
GEN_VEXT_VX(VRemu_vx_32)
GEN_VEXT_VX(VRemu_vx_64)
GEN_VEXT_VX(VRem_vx_8)
GEN_VEXT_VX(VRem_vx_16)
GEN_VEXT_VX(VRem_vx_32)
GEN_VEXT_VX(VRem_vx_64)

static inline uint64_t deposit64(uint64_t value, int start, int length,
                                 uint64_t fieldval)
{
   uint64_t mask;
   vassert(start >= 0 && length > 0 && length <= 64 - start);
   mask = (~0ULL >> (64 - length)) << start;
   return (value & ~mask) | ((fieldval << start) & mask);
}

static inline void vext_set_elem_mask(void *v0, int index,
                                      uint8_t value)
{
   int idx = index / 64;
   int pos = index % 64;
   uint64_t old = ((uint64_t *)v0)[idx];
   ((uint64_t *)v0)[idx] = deposit64(old, pos, 1, value);
}

/* Vector Integer Comparison Instructions */
#define DO_MSEQ(N, M) (N == M)
#define DO_MSNE(N, M) (N != M)
#define DO_MSLT(N, M) (N < M)
#define DO_MSLE(N, M) (N <= M)
#define DO_MSGT(N, M) (N > M)

#define GEN_VEXT_CMP_VV(NAME, ETYPE, DO_OP)                       \
static void h_Iop_##NAME(void *vd, void *vs1, void *vs2, int len) \
{                                                                 \
   for (int i = 0; i < len; ++i) {                                \
      ETYPE s1 = *((ETYPE *)vs1 + i);                             \
      ETYPE s2 = *((ETYPE *)vs2 + i);                             \
      vext_set_elem_mask(vd, i, DO_OP(s2, s1));                   \
   }                                                              \
}

GEN_VEXT_CMP_VV(VMseq_vv_8, uint8_t,  DO_MSEQ)
GEN_VEXT_CMP_VV(VMseq_vv_16, uint16_t, DO_MSEQ)
GEN_VEXT_CMP_VV(VMseq_vv_32, uint32_t, DO_MSEQ)
GEN_VEXT_CMP_VV(VMseq_vv_64, uint64_t, DO_MSEQ)

GEN_VEXT_CMP_VV(VMsne_vv_8, uint8_t,  DO_MSNE)
GEN_VEXT_CMP_VV(VMsne_vv_16, uint16_t, DO_MSNE)
GEN_VEXT_CMP_VV(VMsne_vv_32, uint32_t, DO_MSNE)
GEN_VEXT_CMP_VV(VMsne_vv_64, uint64_t, DO_MSNE)

GEN_VEXT_CMP_VV(VMsltu_vv_8, uint8_t,  DO_MSLT)
GEN_VEXT_CMP_VV(VMsltu_vv_16, uint16_t, DO_MSLT)
GEN_VEXT_CMP_VV(VMsltu_vv_32, uint32_t, DO_MSLT)
GEN_VEXT_CMP_VV(VMsltu_vv_64, uint64_t, DO_MSLT)

GEN_VEXT_CMP_VV(VMslt_vv_8, int8_t,  DO_MSLT)
GEN_VEXT_CMP_VV(VMslt_vv_16, int16_t, DO_MSLT)
GEN_VEXT_CMP_VV(VMslt_vv_32, int32_t, DO_MSLT)
GEN_VEXT_CMP_VV(VMslt_vv_64, int64_t, DO_MSLT)

GEN_VEXT_CMP_VV(VMsleu_vv_8, uint8_t,  DO_MSLE)
GEN_VEXT_CMP_VV(VMsleu_vv_16, uint16_t, DO_MSLE)
GEN_VEXT_CMP_VV(VMsleu_vv_32, uint32_t, DO_MSLE)
GEN_VEXT_CMP_VV(VMsleu_vv_64, uint64_t, DO_MSLE)

GEN_VEXT_CMP_VV(VMsle_vv_8, int8_t,  DO_MSLE)
GEN_VEXT_CMP_VV(VMsle_vv_16, int16_t, DO_MSLE)
GEN_VEXT_CMP_VV(VMsle_vv_32, int32_t, DO_MSLE)
GEN_VEXT_CMP_VV(VMsle_vv_64, int64_t, DO_MSLE)

#define GEN_VEXT_CMP_VX(NAME, ETYPE, DO_OP)                             \
static void h_Iop_##NAME(void *vd, Long s1, void *vs2, int len)         \
{                                                                       \
   for (int i = 0; i < len; ++i) {                                      \
      ETYPE s2 = *((ETYPE *)vs2 + i);                                   \
      vext_set_elem_mask(vd, i,                                         \
              DO_OP(s2, (ETYPE)(Long)s1));                              \
   }                                                                    \
}

GEN_VEXT_CMP_VX(VMseq_vx_8, uint8_t,  DO_MSEQ)
GEN_VEXT_CMP_VX(VMseq_vx_16, uint16_t, DO_MSEQ)
GEN_VEXT_CMP_VX(VMseq_vx_32, uint32_t, DO_MSEQ)
GEN_VEXT_CMP_VX(VMseq_vx_64, uint64_t, DO_MSEQ)

GEN_VEXT_CMP_VX(VMsne_vx_8, uint8_t,  DO_MSNE)
GEN_VEXT_CMP_VX(VMsne_vx_16, uint16_t, DO_MSNE)
GEN_VEXT_CMP_VX(VMsne_vx_32, uint32_t, DO_MSNE)
GEN_VEXT_CMP_VX(VMsne_vx_64, uint64_t, DO_MSNE)

GEN_VEXT_CMP_VX(VMsltu_vx_8, uint8_t,  DO_MSLT)
GEN_VEXT_CMP_VX(VMsltu_vx_16, uint16_t, DO_MSLT)
GEN_VEXT_CMP_VX(VMsltu_vx_32, uint32_t, DO_MSLT)
GEN_VEXT_CMP_VX(VMsltu_vx_64, uint64_t, DO_MSLT)

GEN_VEXT_CMP_VX(VMslt_vx_8, int8_t,  DO_MSLT)
GEN_VEXT_CMP_VX(VMslt_vx_16, int16_t, DO_MSLT)
GEN_VEXT_CMP_VX(VMslt_vx_32, int32_t, DO_MSLT)
GEN_VEXT_CMP_VX(VMslt_vx_64, int64_t, DO_MSLT)

GEN_VEXT_CMP_VX(VMsleu_vx_8, uint8_t,  DO_MSLE)
GEN_VEXT_CMP_VX(VMsleu_vx_16, uint16_t, DO_MSLE)
GEN_VEXT_CMP_VX(VMsleu_vx_32, uint32_t, DO_MSLE)
GEN_VEXT_CMP_VX(VMsleu_vx_64, uint64_t, DO_MSLE)

GEN_VEXT_CMP_VX(VMsle_vx_8, int8_t,  DO_MSLE)
GEN_VEXT_CMP_VX(VMsle_vx_16, int16_t, DO_MSLE)
GEN_VEXT_CMP_VX(VMsle_vx_32, int32_t, DO_MSLE)
GEN_VEXT_CMP_VX(VMsle_vx_64, int64_t, DO_MSLE)

GEN_VEXT_CMP_VX(VMsgtu_vx_8, uint8_t,  DO_MSGT)
GEN_VEXT_CMP_VX(VMsgtu_vx_16, uint16_t, DO_MSGT)
GEN_VEXT_CMP_VX(VMsgtu_vx_32, uint32_t, DO_MSGT)
GEN_VEXT_CMP_VX(VMsgtu_vx_64, uint64_t, DO_MSGT)

GEN_VEXT_CMP_VX(VMsgt_vx_8, int8_t,  DO_MSGT)
GEN_VEXT_CMP_VX(VMsgt_vx_16, int16_t, DO_MSGT)
GEN_VEXT_CMP_VX(VMsgt_vx_32, int32_t, DO_MSGT)
GEN_VEXT_CMP_VX(VMsgt_vx_64, int64_t, DO_MSGT)

/* Vector Single-Width Integer Multiply-Add Instructions */
#define OPIVV3(NAME, TD, T1, T2, TX1, TX2, OP)                \
static void do_##NAME(void *vd, void *vs1, void *vs2, int i)  \
{                                                             \
    TX1 s1 = *((T1 *)vs1 + i);                                \
    TX2 s2 = *((T2 *)vs2 + i);                                \
    TD d = *((TD *)vd + i);                                   \
    *((TD *)vd + i) = OP(s2, s1, d);                          \
}

#define DO_MACC(N, M, D) (M * N + D)
#define DO_NMSAC(N, M, D) (-(M * N) + D)
#define DO_MADD(N, M, D) (M * D + N)
#define DO_NMSUB(N, M, D) (-(M * D) + N)
RVVCALL(OPIVV3, VMacc_vv_8, OP_SSS_B, DO_MACC)
RVVCALL(OPIVV3, VMacc_vv_16, OP_SSS_H, DO_MACC)
RVVCALL(OPIVV3, VMacc_vv_32, OP_SSS_W, DO_MACC)
RVVCALL(OPIVV3, VMacc_vv_64, OP_SSS_D, DO_MACC)
RVVCALL(OPIVV3, VNmsac_vv_8, OP_SSS_B, DO_NMSAC)
RVVCALL(OPIVV3, VNmsac_vv_16, OP_SSS_H, DO_NMSAC)
RVVCALL(OPIVV3, VNmsac_vv_32, OP_SSS_W, DO_NMSAC)
RVVCALL(OPIVV3, VNmsac_vv_64, OP_SSS_D, DO_NMSAC)
RVVCALL(OPIVV3, VMadd_vv_8, OP_SSS_B, DO_MADD)
RVVCALL(OPIVV3, VMadd_vv_16, OP_SSS_H, DO_MADD)
RVVCALL(OPIVV3, VMadd_vv_32, OP_SSS_W, DO_MADD)
RVVCALL(OPIVV3, VMadd_vv_64, OP_SSS_D, DO_MADD)
RVVCALL(OPIVV3, VNmsub_vv_8, OP_SSS_B, DO_NMSUB)
RVVCALL(OPIVV3, VNmsub_vv_16, OP_SSS_H, DO_NMSUB)
RVVCALL(OPIVV3, VNmsub_vv_32, OP_SSS_W, DO_NMSUB)
RVVCALL(OPIVV3, VNmsub_vv_64, OP_SSS_D, DO_NMSUB)
GEN_VEXT_VV(VMacc_vv_8)
GEN_VEXT_VV(VMacc_vv_16)
GEN_VEXT_VV(VMacc_vv_32)
GEN_VEXT_VV(VMacc_vv_64)
GEN_VEXT_VV(VNmsac_vv_8)
GEN_VEXT_VV(VNmsac_vv_16)
GEN_VEXT_VV(VNmsac_vv_32)
GEN_VEXT_VV(VNmsac_vv_64)
GEN_VEXT_VV(VMadd_vv_8)
GEN_VEXT_VV(VMadd_vv_16)
GEN_VEXT_VV(VMadd_vv_32)
GEN_VEXT_VV(VMadd_vv_64)
GEN_VEXT_VV(VNmsub_vv_8)
GEN_VEXT_VV(VNmsub_vv_16)
GEN_VEXT_VV(VNmsub_vv_32)
GEN_VEXT_VV(VNmsub_vv_64)

#define OPIVX3(NAME, TD, T1, T2, TX1, TX2, OP)             \
static void do_##NAME(void *vd, Long s1, void *vs2, int i) \
{                                                          \
    TX2 s2 = *((T2 *)vs2 + i);                             \
    TD d = *((TD *)vd + i);                                \
    *((TD *)vd + i) = OP(s2, (TX1)(T1)s1, d);              \
}

RVVCALL(OPIVX3, VMacc_vx_8, OP_SSS_B, DO_MACC)
RVVCALL(OPIVX3, VMacc_vx_16, OP_SSS_H, DO_MACC)
RVVCALL(OPIVX3, VMacc_vx_32, OP_SSS_W, DO_MACC)
RVVCALL(OPIVX3, VMacc_vx_64, OP_SSS_D, DO_MACC)
RVVCALL(OPIVX3, VNmsac_vx_8, OP_SSS_B, DO_NMSAC)
RVVCALL(OPIVX3, VNmsac_vx_16, OP_SSS_H, DO_NMSAC)
RVVCALL(OPIVX3, VNmsac_vx_32, OP_SSS_W, DO_NMSAC)
RVVCALL(OPIVX3, VNmsac_vx_64, OP_SSS_D, DO_NMSAC)
RVVCALL(OPIVX3, VMadd_vx_8, OP_SSS_B, DO_MADD)
RVVCALL(OPIVX3, VMadd_vx_16, OP_SSS_H, DO_MADD)
RVVCALL(OPIVX3, VMadd_vx_32, OP_SSS_W, DO_MADD)
RVVCALL(OPIVX3, VMadd_vx_64, OP_SSS_D, DO_MADD)
RVVCALL(OPIVX3, VNmsub_vx_8, OP_SSS_B, DO_NMSUB)
RVVCALL(OPIVX3, VNmsub_vx_16, OP_SSS_H, DO_NMSUB)
RVVCALL(OPIVX3, VNmsub_vx_32, OP_SSS_W, DO_NMSUB)
RVVCALL(OPIVX3, VNmsub_vx_64, OP_SSS_D, DO_NMSUB)
GEN_VEXT_VX(VMacc_vx_8)
GEN_VEXT_VX(VMacc_vx_16)
GEN_VEXT_VX(VMacc_vx_32)
GEN_VEXT_VX(VMacc_vx_64)
GEN_VEXT_VX(VNmsac_vx_8)
GEN_VEXT_VX(VNmsac_vx_16)
GEN_VEXT_VX(VNmsac_vx_32)
GEN_VEXT_VX(VNmsac_vx_64)
GEN_VEXT_VX(VMadd_vx_8)
GEN_VEXT_VX(VMadd_vx_16)
GEN_VEXT_VX(VMadd_vx_32)
GEN_VEXT_VX(VMadd_vx_64)
GEN_VEXT_VX(VNmsub_vx_8)
GEN_VEXT_VX(VNmsub_vx_16)
GEN_VEXT_VX(VNmsub_vx_32)
GEN_VEXT_VX(VNmsub_vx_64)

/* Vector Single-Width Bit Shift Instructions */
#define DO_SLL(N, M)  (N << (M))
#define DO_SRL(N, M)  (N >> (M))

/* generate the helpers for shift instructions with two vector operators */
#define GEN_VEXT_SHIFT_VV(NAME, TS1, TS2, OP, MASK)               \
static void h_Iop_##NAME(void *vd, void *vs1, void *vs2, int len) \
{                                                                 \
   for (int i = 0; i < len; ++i) {                                \
      TS1 s1 = *((TS1 *)vs1 + i);                                 \
      TS2 s2 = *((TS2 *)vs2 + i);                                 \
      *((TS1 *)vd + i) = OP(s2, s1 & MASK);                       \
   }                                                              \
}

GEN_VEXT_SHIFT_VV(VSll_vv_8, uint8_t,  uint8_t, DO_SLL, 0x7)
GEN_VEXT_SHIFT_VV(VSll_vv_16, uint16_t, uint16_t, DO_SLL, 0xf)
GEN_VEXT_SHIFT_VV(VSll_vv_32, uint32_t, uint32_t, DO_SLL, 0x1f)
GEN_VEXT_SHIFT_VV(VSll_vv_64, uint64_t, uint64_t, DO_SLL, 0x3f)

GEN_VEXT_SHIFT_VV(VSrl_vv_8, uint8_t, uint8_t, DO_SRL, 0x7)
GEN_VEXT_SHIFT_VV(VSrl_vv_16, uint16_t, uint16_t, DO_SRL, 0xf)
GEN_VEXT_SHIFT_VV(VSrl_vv_32, uint32_t, uint32_t, DO_SRL, 0x1f)
GEN_VEXT_SHIFT_VV(VSrl_vv_64, uint64_t, uint64_t, DO_SRL, 0x3f)

GEN_VEXT_SHIFT_VV(VSra_vv_8, uint8_t,  int8_t, DO_SRL, 0x7)
GEN_VEXT_SHIFT_VV(VSra_vv_16, uint16_t, int16_t, DO_SRL, 0xf)
GEN_VEXT_SHIFT_VV(VSra_vv_32, uint32_t, int32_t, DO_SRL, 0x1f)
GEN_VEXT_SHIFT_VV(VSra_vv_64, uint64_t, int64_t, DO_SRL, 0x3f)

#define GEN_VEXT_SHIFT_VX(NAME, TD, TS2, OP, MASK)                \
static void h_Iop_##NAME(void *vd, Long s1, void *vs2, int len)   \
{                                                                 \
   for (int i = 0; i < len; ++i) {                                \
      TS2 s2 = *((TS2 *)vs2 + i);                                 \
      *((TD *)vd + i) = OP(s2, s1 & MASK);                        \
    }                                                             \
}

GEN_VEXT_SHIFT_VX(VSll_vx_8, uint8_t, int8_t, DO_SLL, 0x7)
GEN_VEXT_SHIFT_VX(VSll_vx_16, uint16_t, int16_t, DO_SLL, 0xf)
GEN_VEXT_SHIFT_VX(VSll_vx_32, uint32_t, int32_t, DO_SLL, 0x1f)
GEN_VEXT_SHIFT_VX(VSll_vx_64, uint64_t, int64_t, DO_SLL, 0x3f)

GEN_VEXT_SHIFT_VX(VSrl_vx_8, uint8_t, uint8_t, DO_SRL, 0x7)
GEN_VEXT_SHIFT_VX(VSrl_vx_16, uint16_t, uint16_t, DO_SRL, 0xf)
GEN_VEXT_SHIFT_VX(VSrl_vx_32, uint32_t, uint32_t, DO_SRL, 0x1f)
GEN_VEXT_SHIFT_VX(VSrl_vx_64, uint64_t, uint64_t, DO_SRL, 0x3f)

GEN_VEXT_SHIFT_VX(VSra_vx_8, int8_t, int8_t, DO_SRL, 0x7)
GEN_VEXT_SHIFT_VX(VSra_vx_16, int16_t, int16_t, DO_SRL, 0xf)
GEN_VEXT_SHIFT_VX(VSra_vx_32, int32_t, int32_t, DO_SRL, 0x1f)
GEN_VEXT_SHIFT_VX(VSra_vx_64, int64_t, int64_t, DO_SRL, 0x3f)

/* Vector Narrowing Integer Right Shift Instructions */
GEN_VEXT_SHIFT_VV(VNsrl_wv_8, uint8_t,  uint16_t, DO_SRL, 0xf)
GEN_VEXT_SHIFT_VV(VNsrl_wv_16, uint16_t, uint32_t, DO_SRL, 0x1f)
GEN_VEXT_SHIFT_VV(VNsrl_wv_32, uint32_t, uint64_t, DO_SRL, 0x3f)
GEN_VEXT_SHIFT_VV(VNsra_wv_8, uint8_t,  int16_t, DO_SRL, 0xf)
GEN_VEXT_SHIFT_VV(VNsra_wv_16, uint16_t, int32_t, DO_SRL, 0x1f)
GEN_VEXT_SHIFT_VV(VNsra_wv_32, uint32_t, int64_t, DO_SRL, 0x3f)
GEN_VEXT_SHIFT_VX(VNsrl_wx_8, uint8_t, uint16_t, DO_SRL, 0xf)
GEN_VEXT_SHIFT_VX(VNsrl_wx_16, uint16_t, uint32_t, DO_SRL, 0x1f)
GEN_VEXT_SHIFT_VX(VNsrl_wx_32, uint32_t, uint64_t, DO_SRL, 0x3f)
GEN_VEXT_SHIFT_VX(VNsra_wx_8, int8_t, int16_t, DO_SRL, 0xf)
GEN_VEXT_SHIFT_VX(VNsra_wx_16, int16_t, int32_t, DO_SRL, 0x1f)
GEN_VEXT_SHIFT_VX(VNsra_wx_32, int32_t, int64_t, DO_SRL, 0x3f)

/* Vector Integer Extension */
#define GEN_VEXT_INT_EXT(NAME, ETYPE, DTYPE)            \
static void h_Iop_##NAME(void *vd, void *vs2, int len)  \
{                                                       \
   for (int i = 0; i < len; ++i) {                      \
      *((ETYPE *)vd + i) = *((DTYPE *)vs2 + i);         \
   }                                                    \
}

GEN_VEXT_INT_EXT(VZext_vf2_16, uint16_t, uint8_t)
GEN_VEXT_INT_EXT(VZext_vf2_32, uint32_t, uint16_t)
GEN_VEXT_INT_EXT(VZext_vf2_64, uint64_t, uint32_t)
GEN_VEXT_INT_EXT(VZext_vf4_32, uint32_t, uint8_t)
GEN_VEXT_INT_EXT(VZext_vf4_64, uint64_t, uint16_t)
GEN_VEXT_INT_EXT(VZext_vf8_64, uint64_t, uint8_t)

GEN_VEXT_INT_EXT(VSext_vf2_16, int16_t, int8_t)
GEN_VEXT_INT_EXT(VSext_vf2_32, int32_t, int16_t)
GEN_VEXT_INT_EXT(VSext_vf2_64, int64_t, int32_t)
GEN_VEXT_INT_EXT(VSext_vf4_32, int32_t, int8_t)
GEN_VEXT_INT_EXT(VSext_vf4_64, int64_t, int16_t)
GEN_VEXT_INT_EXT(VSext_vf8_64, int64_t, int8_t)

/* Vector Widening Integer Add/Subtract */
#define WOP_UUU_B uint16_t, uint8_t, uint8_t, uint16_t, uint16_t
#define WOP_UUU_H uint32_t, uint16_t, uint16_t, uint32_t, uint32_t
#define WOP_UUU_W uint64_t, uint32_t, uint32_t, uint64_t, uint64_t
#define WOP_SSS_B int16_t, int8_t, int8_t, int16_t, int16_t
#define WOP_SSS_H int32_t, int16_t, int16_t, int32_t, int32_t
#define WOP_SSS_W int64_t, int32_t, int32_t, int64_t, int64_t
#define WOP_SUS_B int16_t, uint8_t, int8_t, uint16_t, int16_t
#define WOP_SUS_H int32_t, uint16_t, int16_t, uint32_t, int32_t
#define WOP_SUS_W int64_t, uint32_t, int32_t, uint64_t, int64_t
#define WOP_SSU_B int16_t, int8_t, uint8_t, int16_t, uint16_t
#define WOP_SSU_H int32_t, int16_t, uint16_t, int32_t, uint32_t
#define WOP_SSU_W int64_t, int32_t, uint32_t, int64_t, uint64_t
#define WOP_WUUU_B  uint16_t, uint8_t, uint16_t, uint16_t, uint16_t
#define WOP_WUUU_H  uint32_t, uint16_t, uint32_t, uint32_t, uint32_t
#define WOP_WUUU_W  uint64_t, uint32_t, uint64_t, uint64_t, uint64_t
#define WOP_WSSS_B  int16_t, int8_t, int16_t, int16_t, int16_t
#define WOP_WSSS_H  int32_t, int16_t, int32_t, int32_t, int32_t
#define WOP_WSSS_W  int64_t, int32_t, int64_t, int64_t, int64_t
RVVCALL(OPIVV2, VWaddu_vv_8, WOP_UUU_B, DO_ADD)
RVVCALL(OPIVV2, VWaddu_vv_16, WOP_UUU_H, DO_ADD)
RVVCALL(OPIVV2, VWaddu_vv_32, WOP_UUU_W, DO_ADD)
RVVCALL(OPIVV2, VWsubu_vv_8, WOP_UUU_B, DO_SUB)
RVVCALL(OPIVV2, VWsubu_vv_16, WOP_UUU_H, DO_SUB)
RVVCALL(OPIVV2, VWsubu_vv_32, WOP_UUU_W, DO_SUB)
RVVCALL(OPIVV2, VWadd_vv_8, WOP_SSS_B, DO_ADD)
RVVCALL(OPIVV2, VWadd_vv_16, WOP_SSS_H, DO_ADD)
RVVCALL(OPIVV2, VWadd_vv_32, WOP_SSS_W, DO_ADD)
RVVCALL(OPIVV2, VWsub_vv_8, WOP_SSS_B, DO_SUB)
RVVCALL(OPIVV2, VWsub_vv_16, WOP_SSS_H, DO_SUB)
RVVCALL(OPIVV2, VWsub_vv_32, WOP_SSS_W, DO_SUB)
RVVCALL(OPIVV2, VWaddu_wv_8, WOP_WUUU_B, DO_ADD)
RVVCALL(OPIVV2, VWaddu_wv_16, WOP_WUUU_H, DO_ADD)
RVVCALL(OPIVV2, VWaddu_wv_32, WOP_WUUU_W, DO_ADD)
RVVCALL(OPIVV2, VWsubu_wv_8, WOP_WUUU_B, DO_SUB)
RVVCALL(OPIVV2, VWsubu_wv_16, WOP_WUUU_H, DO_SUB)
RVVCALL(OPIVV2, VWsubu_wv_32, WOP_WUUU_W, DO_SUB)
RVVCALL(OPIVV2, VWadd_wv_8, WOP_WSSS_B, DO_ADD)
RVVCALL(OPIVV2, VWadd_wv_16, WOP_WSSS_H, DO_ADD)
RVVCALL(OPIVV2, VWadd_wv_32, WOP_WSSS_W, DO_ADD)
RVVCALL(OPIVV2, VWsub_wv_8, WOP_WSSS_B, DO_SUB)
RVVCALL(OPIVV2, VWsub_wv_16, WOP_WSSS_H, DO_SUB)
RVVCALL(OPIVV2, VWsub_wv_32, WOP_WSSS_W, DO_SUB)
GEN_VEXT_VV(VWaddu_vv_8)
GEN_VEXT_VV(VWaddu_vv_16)
GEN_VEXT_VV(VWaddu_vv_32)
GEN_VEXT_VV(VWsubu_vv_8)
GEN_VEXT_VV(VWsubu_vv_16)
GEN_VEXT_VV(VWsubu_vv_32)
GEN_VEXT_VV(VWadd_vv_8)
GEN_VEXT_VV(VWadd_vv_16)
GEN_VEXT_VV(VWadd_vv_32)
GEN_VEXT_VV(VWsub_vv_8)
GEN_VEXT_VV(VWsub_vv_16)
GEN_VEXT_VV(VWsub_vv_32)
GEN_VEXT_VV(VWaddu_wv_8)
GEN_VEXT_VV(VWaddu_wv_16)
GEN_VEXT_VV(VWaddu_wv_32)
GEN_VEXT_VV(VWsubu_wv_8)
GEN_VEXT_VV(VWsubu_wv_16)
GEN_VEXT_VV(VWsubu_wv_32)
GEN_VEXT_VV(VWadd_wv_8)
GEN_VEXT_VV(VWadd_wv_16)
GEN_VEXT_VV(VWadd_wv_32)
GEN_VEXT_VV(VWsub_wv_8)
GEN_VEXT_VV(VWsub_wv_16)
GEN_VEXT_VV(VWsub_wv_32)

RVVCALL(OPIVX2, VWaddu_vx_8, WOP_UUU_B, DO_ADD)
RVVCALL(OPIVX2, VWaddu_vx_16, WOP_UUU_H, DO_ADD)
RVVCALL(OPIVX2, VWaddu_vx_32, WOP_UUU_W, DO_ADD)
RVVCALL(OPIVX2, VWsubu_vx_8, WOP_UUU_B, DO_SUB)
RVVCALL(OPIVX2, VWsubu_vx_16, WOP_UUU_H, DO_SUB)
RVVCALL(OPIVX2, VWsubu_vx_32, WOP_UUU_W, DO_SUB)
RVVCALL(OPIVX2, VWadd_vx_8, WOP_SSS_B, DO_ADD)
RVVCALL(OPIVX2, VWadd_vx_16, WOP_SSS_H, DO_ADD)
RVVCALL(OPIVX2, VWadd_vx_32, WOP_SSS_W, DO_ADD)
RVVCALL(OPIVX2, VWsub_vx_8, WOP_SSS_B, DO_SUB)
RVVCALL(OPIVX2, VWsub_vx_16, WOP_SSS_H, DO_SUB)
RVVCALL(OPIVX2, VWsub_vx_32, WOP_SSS_W, DO_SUB)
RVVCALL(OPIVX2, VWaddu_wx_8, WOP_WUUU_B, DO_ADD)
RVVCALL(OPIVX2, VWaddu_wx_16, WOP_WUUU_H, DO_ADD)
RVVCALL(OPIVX2, VWaddu_wx_32, WOP_WUUU_W, DO_ADD)
RVVCALL(OPIVX2, VWsubu_wx_8, WOP_WUUU_B, DO_SUB)
RVVCALL(OPIVX2, VWsubu_wx_16, WOP_WUUU_H, DO_SUB)
RVVCALL(OPIVX2, VWsubu_wx_32, WOP_WUUU_W, DO_SUB)
RVVCALL(OPIVX2, VWadd_wx_8, WOP_WSSS_B, DO_ADD)
RVVCALL(OPIVX2, VWadd_wx_16, WOP_WSSS_H, DO_ADD)
RVVCALL(OPIVX2, VWadd_wx_32, WOP_WSSS_W, DO_ADD)
RVVCALL(OPIVX2, VWsub_wx_8, WOP_WSSS_B, DO_SUB)
RVVCALL(OPIVX2, VWsub_wx_16, WOP_WSSS_H, DO_SUB)
RVVCALL(OPIVX2, VWsub_wx_32, WOP_WSSS_W, DO_SUB)
GEN_VEXT_VX(VWaddu_vx_8)
GEN_VEXT_VX(VWaddu_vx_16)
GEN_VEXT_VX(VWaddu_vx_32)
GEN_VEXT_VX(VWsubu_vx_8)
GEN_VEXT_VX(VWsubu_vx_16)
GEN_VEXT_VX(VWsubu_vx_32)
GEN_VEXT_VX(VWadd_vx_8)
GEN_VEXT_VX(VWadd_vx_16)
GEN_VEXT_VX(VWadd_vx_32)
GEN_VEXT_VX(VWsub_vx_8)
GEN_VEXT_VX(VWsub_vx_16)
GEN_VEXT_VX(VWsub_vx_32)
GEN_VEXT_VX(VWaddu_wx_8)
GEN_VEXT_VX(VWaddu_wx_16)
GEN_VEXT_VX(VWaddu_wx_32)
GEN_VEXT_VX(VWsubu_wx_8)
GEN_VEXT_VX(VWsubu_wx_16)
GEN_VEXT_VX(VWsubu_wx_32)
GEN_VEXT_VX(VWadd_wx_8)
GEN_VEXT_VX(VWadd_wx_16)
GEN_VEXT_VX(VWadd_wx_32)
GEN_VEXT_VX(VWsub_wx_8)
GEN_VEXT_VX(VWsub_wx_16)
GEN_VEXT_VX(VWsub_wx_32)

/* Vector Widening Integer Multiply Instructions */
RVVCALL(OPIVV2, VWmul_vv_8, WOP_SSS_B, DO_MUL)
RVVCALL(OPIVV2, VWmul_vv_16, WOP_SSS_H, DO_MUL)
RVVCALL(OPIVV2, VWmul_vv_32, WOP_SSS_W, DO_MUL)
RVVCALL(OPIVV2, VWmulu_vv_8, WOP_UUU_B, DO_MUL)
RVVCALL(OPIVV2, VWmulu_vv_16, WOP_UUU_H, DO_MUL)
RVVCALL(OPIVV2, VWmulu_vv_32, WOP_UUU_W, DO_MUL)
RVVCALL(OPIVV2, VWmulsu_vv_8, WOP_SUS_B, DO_MUL)
RVVCALL(OPIVV2, VWmulsu_vv_16, WOP_SUS_H, DO_MUL)
RVVCALL(OPIVV2, VWmulsu_vv_32, WOP_SUS_W, DO_MUL)
GEN_VEXT_VV(VWmul_vv_8)
GEN_VEXT_VV(VWmul_vv_16)
GEN_VEXT_VV(VWmul_vv_32)
GEN_VEXT_VV(VWmulu_vv_8)
GEN_VEXT_VV(VWmulu_vv_16)
GEN_VEXT_VV(VWmulu_vv_32)
GEN_VEXT_VV(VWmulsu_vv_8)
GEN_VEXT_VV(VWmulsu_vv_16)
GEN_VEXT_VV(VWmulsu_vv_32)

RVVCALL(OPIVX2, VWmul_vx_8, WOP_SSS_B, DO_MUL)
RVVCALL(OPIVX2, VWmul_vx_16, WOP_SSS_H, DO_MUL)
RVVCALL(OPIVX2, VWmul_vx_32, WOP_SSS_W, DO_MUL)
RVVCALL(OPIVX2, VWmulu_vx_8, WOP_UUU_B, DO_MUL)
RVVCALL(OPIVX2, VWmulu_vx_16, WOP_UUU_H, DO_MUL)
RVVCALL(OPIVX2, VWmulu_vx_32, WOP_UUU_W, DO_MUL)
RVVCALL(OPIVX2, VWmulsu_vx_8, WOP_SUS_B, DO_MUL)
RVVCALL(OPIVX2, VWmulsu_vx_16, WOP_SUS_H, DO_MUL)
RVVCALL(OPIVX2, VWmulsu_vx_32, WOP_SUS_W, DO_MUL)
GEN_VEXT_VX(VWmul_vx_8)
GEN_VEXT_VX(VWmul_vx_16)
GEN_VEXT_VX(VWmul_vx_32)
GEN_VEXT_VX(VWmulu_vx_8)
GEN_VEXT_VX(VWmulu_vx_16)
GEN_VEXT_VX(VWmulu_vx_32)
GEN_VEXT_VX(VWmulsu_vx_8)
GEN_VEXT_VX(VWmulsu_vx_16)
GEN_VEXT_VX(VWmulsu_vx_32)

/* Vector Widening Integer Multiply-Add Instructions */
RVVCALL(OPIVV3, VWmaccu_vv_8, WOP_UUU_B, DO_MACC)
RVVCALL(OPIVV3, VWmaccu_vv_16, WOP_UUU_H, DO_MACC)
RVVCALL(OPIVV3, VWmaccu_vv_32, WOP_UUU_W, DO_MACC)
RVVCALL(OPIVV3, VWmacc_vv_8, WOP_SSS_B, DO_MACC)
RVVCALL(OPIVV3, VWmacc_vv_16, WOP_SSS_H, DO_MACC)
RVVCALL(OPIVV3, VWmacc_vv_32, WOP_SSS_W, DO_MACC)
RVVCALL(OPIVV3, VWmaccsu_vv_8, WOP_SSU_B, DO_MACC)
RVVCALL(OPIVV3, VWmaccsu_vv_16, WOP_SSU_H, DO_MACC)
RVVCALL(OPIVV3, VWmaccsu_vv_32, WOP_SSU_W, DO_MACC)
GEN_VEXT_VV(VWmaccu_vv_8)
GEN_VEXT_VV(VWmaccu_vv_16)
GEN_VEXT_VV(VWmaccu_vv_32)
GEN_VEXT_VV(VWmacc_vv_8)
GEN_VEXT_VV(VWmacc_vv_16)
GEN_VEXT_VV(VWmacc_vv_32)
GEN_VEXT_VV(VWmaccsu_vv_8)
GEN_VEXT_VV(VWmaccsu_vv_16)
GEN_VEXT_VV(VWmaccsu_vv_32)

RVVCALL(OPIVX3, VWmaccu_vx_8, WOP_UUU_B, DO_MACC)
RVVCALL(OPIVX3, VWmaccu_vx_16, WOP_UUU_H, DO_MACC)
RVVCALL(OPIVX3, VWmaccu_vx_32, WOP_UUU_W, DO_MACC)
RVVCALL(OPIVX3, VWmacc_vx_8, WOP_SSS_B, DO_MACC)
RVVCALL(OPIVX3, VWmacc_vx_16, WOP_SSS_H, DO_MACC)
RVVCALL(OPIVX3, VWmacc_vx_32, WOP_SSS_W, DO_MACC)
RVVCALL(OPIVX3, VWmaccsu_vx_8, WOP_SSU_B, DO_MACC)
RVVCALL(OPIVX3, VWmaccsu_vx_16, WOP_SSU_H, DO_MACC)
RVVCALL(OPIVX3, VWmaccsu_vx_32, WOP_SSU_W, DO_MACC)
RVVCALL(OPIVX3, VWmaccus_vx_8, WOP_SUS_B, DO_MACC)
RVVCALL(OPIVX3, VWmaccus_vx_16, WOP_SUS_H, DO_MACC)
RVVCALL(OPIVX3, VWmaccus_vx_32, WOP_SUS_W, DO_MACC)
GEN_VEXT_VX(VWmaccu_vx_8)
GEN_VEXT_VX(VWmaccu_vx_16)
GEN_VEXT_VX(VWmaccu_vx_32)
GEN_VEXT_VX(VWmacc_vx_8)
GEN_VEXT_VX(VWmacc_vx_16)
GEN_VEXT_VX(VWmacc_vx_32)
GEN_VEXT_VX(VWmaccsu_vx_8)
GEN_VEXT_VX(VWmaccsu_vx_16)
GEN_VEXT_VX(VWmaccsu_vx_32)
GEN_VEXT_VX(VWmaccus_vx_8)
GEN_VEXT_VX(VWmaccus_vx_16)
GEN_VEXT_VX(VWmaccus_vx_32)

#define GEN_VEXT_VMV_VV(NAME, ETYPE)                              \
static void h_Iop_##NAME(void *vd, void *vs1, int len)            \
{                                                                 \
   for (int i = 0; i < len; ++i) {                                \
        ETYPE s1 = *((ETYPE *)vs1 + i);                           \
        *((ETYPE *)vd + i) = s1;                                  \
    }                                                             \
}

GEN_VEXT_VMV_VV(VMv_v_v_8, int8_t)
GEN_VEXT_VMV_VV(VMv_v_v_16, int16_t)
GEN_VEXT_VMV_VV(VMv_v_v_32, int32_t)
GEN_VEXT_VMV_VV(VMv_v_v_64, int64_t)

#define GEN_VEXT_VMV_VX(NAME, ETYPE)                              \
static void h_Iop_##NAME(void *vd, uint64_t s1, int len)          \
{                                                                 \
    for (int i = 0; i < len; ++i) {                               \
        *((ETYPE *)vd + i) = (ETYPE)s1;                           \
    }                                                             \
}

GEN_VEXT_VMV_VX(VMv_v_x_8, int8_t)
GEN_VEXT_VMV_VX(VMv_v_x_16, int16_t)
GEN_VEXT_VMV_VX(VMv_v_x_32, int32_t)
GEN_VEXT_VMV_VX(VMv_v_x_64, int64_t)

static inline int vext_elem_mask(void *v0, int index)
{
    int idx = index / 64;
    int pos = index  % 64;
    return (((uint64_t *)v0)[idx] >> pos) & 1;
}

#define GEN_VEXT_VMERGE_VV(NAME, ETYPE)                                      \
static void h_Iop_##NAME(void *vd, void *vs1, void *vs2, void *v0, int len)  \
{                                                                            \
    for (int i = 0; i < len; ++i) {                                          \
        ETYPE *vt = (!vext_elem_mask(v0, i) ? vs2 : vs1);                    \
        *((ETYPE *)vd + i) = *(vt + i);                                      \
    }                                                                        \
}

GEN_VEXT_VMERGE_VV(VMerge_vvm_8, int8_t)
GEN_VEXT_VMERGE_VV(VMerge_vvm_16, int16_t)
GEN_VEXT_VMERGE_VV(VMerge_vvm_32, int32_t)
GEN_VEXT_VMERGE_VV(VMerge_vvm_64, int64_t)

#define GEN_VEXT_VMERGE_VX(NAME, ETYPE)                                      \
static void h_Iop_##NAME(void *vd, Long s1, void *vs2, void *v0, int len)    \
{                                                                            \
    for (int i = 0; i < len; ++i) {                                          \
        ETYPE s2 = *((ETYPE *)vs2 + i);                                      \
        ETYPE d = (!vext_elem_mask(v0, i) ? s2 : (ETYPE)s1);                 \
        *((ETYPE *)vd + i) = d;                                              \
    }                                                                        \
}

GEN_VEXT_VMERGE_VX(VMerge_vxm_8, int8_t)
GEN_VEXT_VMERGE_VX(VMerge_vxm_16, int16_t)
GEN_VEXT_VMERGE_VX(VMerge_vxm_32, int32_t)
GEN_VEXT_VMERGE_VX(VMerge_vxm_64, int64_t)

/* Vector Single-Width Integer Reduction Instructions */
#define GEN_VEXT_RED_NOM(NAME, SFX, TD, TS2, OP)                           \
static void h_Iop_##NAME##_##SFX(void *vd, Long vs1_0, void *vs2, int len) \
{                                                                          \
   TD s1 = (TD)vs1_0;                                                      \
                                                                           \
   for (int i = 0; i < len; ++i) {                                         \
      TS2 s2 = *((TS2 *)vs2 + i);                                          \
      s1 = OP(s1, (TD)s2);                                                 \
   }                                                                       \
   *((TD *)vd + 0) = s1;                                                   \
}

#define GEN_VEXT_RED_M(NAME, SFX, TD, TS2, OP)                             \
static void h_Iop_##NAME##m_##SFX(void *vd, Long vs1_0, void *vs2,         \
                                  void *v0, int len)                       \
{                                                                          \
   TD s1 = (TD)vs1_0;                                                      \
                                                                           \
   for (int i = 0; i < len; ++i) {                                         \
      if (!vext_elem_mask(v0, i)) {                                        \
         continue;                                                         \
      }                                                                    \
      TS2 s2 = *((TS2 *)vs2 + i);                                          \
      s1 = OP(s1, (TD)s2);                                                 \
   }                                                                       \
   *((TD *)vd + 0) = s1;                                                   \
}

#define GEN_VEXT_RED(NAME, SFX, TD, TS2, OP) \
   GEN_VEXT_RED_M(NAME, SFX, TD, TS2, OP)    \
   GEN_VEXT_RED_NOM(NAME, SFX, TD, TS2, OP)

/* vd[0] = sum(vs1[0], vs2[*]) */
GEN_VEXT_RED(VRedsum_vs, 8, int8_t,  int8_t,  DO_ADD)
GEN_VEXT_RED(VRedsum_vs, 16, int16_t, int16_t, DO_ADD)
GEN_VEXT_RED(VRedsum_vs, 32, int32_t, int32_t, DO_ADD)
GEN_VEXT_RED(VRedsum_vs, 64, int64_t, int64_t, DO_ADD)

/* vd[0] = maxu(vs1[0], vs2[*]) */
GEN_VEXT_RED(VRedmaxu_vs, 8, uint8_t,  uint8_t,  DO_MAX)
GEN_VEXT_RED(VRedmaxu_vs, 16, uint16_t, uint16_t, DO_MAX)
GEN_VEXT_RED(VRedmaxu_vs, 32, uint32_t, uint32_t, DO_MAX)
GEN_VEXT_RED(VRedmaxu_vs, 64, uint64_t, uint64_t, DO_MAX)

/* vd[0] = max(vs1[0], vs2[*]) */
GEN_VEXT_RED(VRedmax_vs, 8, int8_t,  int8_t,  DO_MAX)
GEN_VEXT_RED(VRedmax_vs, 16, int16_t, int16_t, DO_MAX)
GEN_VEXT_RED(VRedmax_vs, 32, int32_t, int32_t, DO_MAX)
GEN_VEXT_RED(VRedmax_vs, 64, int64_t, int64_t, DO_MAX)

/* vd[0] = minu(vs1[0], vs2[*]) */
GEN_VEXT_RED(VRedminu_vs, 8, uint8_t,  uint8_t,  DO_MIN)
GEN_VEXT_RED(VRedminu_vs, 16, uint16_t, uint16_t, DO_MIN)
GEN_VEXT_RED(VRedminu_vs, 32, uint32_t, uint32_t, DO_MIN)
GEN_VEXT_RED(VRedminu_vs, 64, uint64_t, uint64_t, DO_MIN)

/* vd[0] = min(vs1[0], vs2[*]) */
GEN_VEXT_RED(VRedmin_vs, 8, int8_t,  int8_t,  DO_MIN)
GEN_VEXT_RED(VRedmin_vs, 16, int16_t, int16_t, DO_MIN)
GEN_VEXT_RED(VRedmin_vs, 32, int32_t, int32_t, DO_MIN)
GEN_VEXT_RED(VRedmin_vs, 64, int64_t, int64_t, DO_MIN)

/* vd[0] = and(vs1[0], vs2[*]) */
GEN_VEXT_RED(VRedand_vs, 8, int8_t,  int8_t,  DO_AND)
GEN_VEXT_RED(VRedand_vs, 16, int16_t, int16_t, DO_AND)
GEN_VEXT_RED(VRedand_vs, 32, int32_t, int32_t, DO_AND)
GEN_VEXT_RED(VRedand_vs, 64, int64_t, int64_t, DO_AND)

/* vd[0] = or(vs1[0], vs2[*]) */
GEN_VEXT_RED(VRedor_vs, 8, int8_t,  int8_t,  DO_OR)
GEN_VEXT_RED(VRedor_vs, 16, int16_t, int16_t, DO_OR)
GEN_VEXT_RED(VRedor_vs, 32, int32_t, int32_t, DO_OR)
GEN_VEXT_RED(VRedor_vs, 64, int64_t, int64_t, DO_OR)

/* vd[0] = xor(vs1[0], vs2[*]) */
GEN_VEXT_RED(VRedxor_vs, 8, int8_t,  int8_t,  DO_XOR)
GEN_VEXT_RED(VRedxor_vs, 16, int16_t, int16_t, DO_XOR)
GEN_VEXT_RED(VRedxor_vs, 32, int32_t, int32_t, DO_XOR)
GEN_VEXT_RED(VRedxor_vs, 64, int64_t, int64_t, DO_XOR)

/* Vector Widening Integer Reduction Instructions */
/* signed sum reduction into double-width accumulator */
GEN_VEXT_RED(VWredsum_vs, 8, int16_t, int8_t, DO_ADD)
GEN_VEXT_RED(VWredsum_vs, 16, int32_t, int16_t, DO_ADD)
GEN_VEXT_RED(VWredsum_vs, 32, int64_t, int32_t, DO_ADD)

/* Unsigned sum reduction into double-width accumulator */
GEN_VEXT_RED(VWredsumu_vs, 8, uint16_t, uint8_t, DO_ADD)
GEN_VEXT_RED(VWredsumu_vs, 16, uint32_t, uint16_t, DO_ADD)
GEN_VEXT_RED(VWredsumu_vs, 32, uint64_t, uint32_t, DO_ADD)

/*
 * Vector Mask Operations
 */
/* Vector Mask-Register Logical Instructions */
#define GEN_VEXT_MASK_VV(NAME, OP)                        \
static void h_Iop_##NAME(void *vd, void *vs1, void *vs2,  \
                  int len)                                \
{                                                         \
    int a, b;                                             \
                                                          \
    for (int i = 0; i < len; ++i) {                       \
        a = vext_elem_mask(vs1, i);                       \
        b = vext_elem_mask(vs2, i);                       \
        vext_set_elem_mask(vd, i, OP(b, a));              \
    }                                                     \
}

#define DO_NAND(N, M)  (!(N & M))
#define DO_ANDNOT(N, M)  (N & !M)
#define DO_NOR(N, M)  (!(N | M))
#define DO_ORNOT(N, M)  (N | !M)
#define DO_XNOR(N, M)  (!(N ^ M))

GEN_VEXT_MASK_VV(VMand_mm, DO_AND)
GEN_VEXT_MASK_VV(VMnand_mm, DO_NAND)
GEN_VEXT_MASK_VV(VMandn_mm, DO_ANDNOT)
GEN_VEXT_MASK_VV(VMxor_mm, DO_XOR)
GEN_VEXT_MASK_VV(VMor_mm, DO_OR)
GEN_VEXT_MASK_VV(VMnor_mm, DO_NOR)
GEN_VEXT_MASK_VV(VMorn_mm, DO_ORNOT)
GEN_VEXT_MASK_VV(VMxnor_mm, DO_XNOR)

/* Vector Integer Add-with-Carry / Subtract-with-Borrow Instructions */
#define DO_VADC(N, M, C) (N + M + C)
#define DO_VSBC(N, M, C) (N - M - C)

#define GEN_VEXT_VADC_VVM(NAME, ETYPE, DO_OP)                 \
static void h_Iop_##NAME(void *vd, void *vs1, void *vs2,      \
                         void *v0, int len)                   \
{                                                             \
    for (int i = 0; i < len; ++i) {                           \
        ETYPE s1 = *((ETYPE *)vs1 + i);                       \
        ETYPE s2 = *((ETYPE *)vs2 + i);                       \
        ETYPE carry = vext_elem_mask(v0, i);                  \
                                                              \
        *((ETYPE *)vd + i) = DO_OP(s2, s1, carry);            \
    }                                                         \
}

GEN_VEXT_VADC_VVM(VAdc_vvm_8, uint8_t, DO_VADC)
GEN_VEXT_VADC_VVM(VAdc_vvm_16, uint16_t, DO_VADC)
GEN_VEXT_VADC_VVM(VAdc_vvm_32, uint32_t, DO_VADC)
GEN_VEXT_VADC_VVM(VAdc_vvm_64, uint64_t, DO_VADC)

GEN_VEXT_VADC_VVM(VSbc_vvm_8, uint8_t, DO_VSBC)
GEN_VEXT_VADC_VVM(VSbc_vvm_16, uint16_t, DO_VSBC)
GEN_VEXT_VADC_VVM(VSbc_vvm_32, uint32_t, DO_VSBC)
GEN_VEXT_VADC_VVM(VSbc_vvm_64, uint64_t, DO_VSBC)

#define GEN_VEXT_VADC_VXM(NAME, ETYPE, DO_OP)                     \
static void h_Iop_##NAME(void *vd, Long s1, void *vs2, void *v0,  \
                         int len)                                 \
{                                                                 \
    for (int i = 0; i < len; ++i) {                               \
        ETYPE s2 = *((ETYPE *)vs2 + i);                           \
        ETYPE carry = vext_elem_mask(v0, i);                      \
                                                                  \
        *((ETYPE *)vd + i) = DO_OP(s2, (ETYPE)(Long)s1, carry);   \
    }                                                             \
}

GEN_VEXT_VADC_VXM(VAdc_vxm_8, uint8_t, DO_VADC)
GEN_VEXT_VADC_VXM(VAdc_vxm_16, uint16_t, DO_VADC)
GEN_VEXT_VADC_VXM(VAdc_vxm_32, uint32_t, DO_VADC)
GEN_VEXT_VADC_VXM(VAdc_vxm_64, uint64_t, DO_VADC)

GEN_VEXT_VADC_VXM(VSbc_vxm_8, uint8_t, DO_VSBC)
GEN_VEXT_VADC_VXM(VSbc_vxm_16, uint16_t, DO_VSBC)
GEN_VEXT_VADC_VXM(VSbc_vxm_32, uint32_t, DO_VSBC)
GEN_VEXT_VADC_VXM(VSbc_vxm_64, uint64_t, DO_VSBC)

#define DO_MADC(N, M, C) (C ? (__typeof(N))(N + M + 1) <= N :           \
                          (__typeof(N))(N + M) < N)
#define DO_MSBC(N, M, C) (C ? N <= M : N < M)

#define GEN_VEXT_VMADC_VV_M(NAME, SFX, ETYPE, DO_OP)              \
static void h_Iop_##NAME##m_##SFX(void *vd, void *vs1, void *vs2, \
                                  void *v0, int len)              \
{                                                                 \
    for (int i = 0; i < len; ++i) {                               \
        ETYPE s1 = *((ETYPE *)vs1 + i);                           \
        ETYPE s2 = *((ETYPE *)vs2 + i);                           \
        ETYPE carry = vext_elem_mask(v0, i);                      \
        vext_set_elem_mask(vd, i, DO_OP(s2, s1, carry));          \
    }                                                             \
}

#define GEN_VEXT_VMADC_VV_NOM(NAME, SFX, ETYPE, DO_OP)            \
static void h_Iop_##NAME##_##SFX(void *vd, void *vs1, void *vs2,  \
                                 int len)                         \
{                                                                 \
    for (int i = 0; i < len; ++i) {                               \
        ETYPE s1 = *((ETYPE *)vs1 + i);                           \
        ETYPE s2 = *((ETYPE *)vs2 + i);                           \
        ETYPE carry = 0;                                          \
        vext_set_elem_mask(vd, i, DO_OP(s2, s1, carry));          \
    }                                                             \
}

#define GEN_VEXT_VMADC_VVM(NAME, SFX, ETYPE, DO_OP)  \
   GEN_VEXT_VMADC_VV_M(NAME, SFX, ETYPE, DO_OP)      \
   GEN_VEXT_VMADC_VV_NOM(NAME, SFX, ETYPE, DO_OP)    \

GEN_VEXT_VMADC_VVM(VMadc_vv, 8, uint8_t, DO_MADC)
GEN_VEXT_VMADC_VVM(VMadc_vv, 16, uint16_t, DO_MADC)
GEN_VEXT_VMADC_VVM(VMadc_vv, 32, uint32_t, DO_MADC)
GEN_VEXT_VMADC_VVM(VMadc_vv, 64, uint64_t, DO_MADC)

GEN_VEXT_VMADC_VVM(VMsbc_vv, 8, uint8_t, DO_MSBC)
GEN_VEXT_VMADC_VVM(VMsbc_vv, 16, uint16_t, DO_MSBC)
GEN_VEXT_VMADC_VVM(VMsbc_vv, 32, uint32_t, DO_MSBC)
GEN_VEXT_VMADC_VVM(VMsbc_vv, 64, uint64_t, DO_MSBC)

#define GEN_VEXT_VMADC_VX_M(NAME, SFX, ETYPE, DO_OP)              \
static void h_Iop_##NAME##m_##SFX(void *vd, Long s1, void *vs2,   \
                                  void *v0, int len)              \
{                                                                 \
    for (int i = 0; i < len; ++i) {                               \
        ETYPE s2 = *((ETYPE *)vs2 + i);                           \
        ETYPE carry = vext_elem_mask(v0, i);                      \
        vext_set_elem_mask(vd, i,                                 \
                DO_OP(s2, (ETYPE)(Long)s1, carry));               \
    }                                                             \
}

#define GEN_VEXT_VMADC_VX_NOM(NAME, SFX, ETYPE, DO_OP)            \
static void h_Iop_##NAME##_##SFX(void *vd, Long s1, void *vs2,    \
                                 int len)                         \
{                                                                 \
    for (int i = 0; i < len; ++i) {                               \
        ETYPE s2 = *((ETYPE *)vs2 + i);                           \
        ETYPE carry = 0;                                          \
        vext_set_elem_mask(vd, i,                                 \
                DO_OP(s2, (ETYPE)(Long)s1, carry));               \
    }                                                             \
}

#define GEN_VEXT_VMADC_VXM(NAME, SFX, ETYPE, DO_OP)  \
   GEN_VEXT_VMADC_VX_M(NAME, SFX, ETYPE, DO_OP)      \
   GEN_VEXT_VMADC_VX_NOM(NAME, SFX, ETYPE, DO_OP)

GEN_VEXT_VMADC_VXM(VMadc_vx, 8, uint8_t, DO_MADC)
GEN_VEXT_VMADC_VXM(VMadc_vx, 16, uint16_t, DO_MADC)
GEN_VEXT_VMADC_VXM(VMadc_vx, 32, uint32_t, DO_MADC)
GEN_VEXT_VMADC_VXM(VMadc_vx, 64, uint64_t, DO_MADC)

GEN_VEXT_VMADC_VXM(VMsbc_vx, 8, uint8_t, DO_MSBC)
GEN_VEXT_VMADC_VXM(VMsbc_vx, 16, uint16_t, DO_MSBC)
GEN_VEXT_VMADC_VXM(VMsbc_vx, 32, uint32_t, DO_MSBC)
GEN_VEXT_VMADC_VXM(VMsbc_vx, 64, uint64_t, DO_MSBC)

/* Vector count population in mask vcpop */
static void h_Iop_VCpop_m(void *dst, void *vs2, int len)
{
    ULong cnt = 0;
    for (int i = 0; i < len; ++i) {
        if (vext_elem_mask(vs2, i)) {
           cnt++;
        }
    }

    *(ULong *)dst = cnt;
}

/* vfirst find-first-set mask bit */
static void h_Iop_VFirst_m(void *dst, void *vs2, int len)
{
    ULong first = -1ULL;
    for (int i = 0; i < len; ++i) {
         if (vext_elem_mask(vs2, i)) {
             first = i;
             break;
         }
    }

    *(ULong *)dst = first;
}

enum set_mask_type {
    ONLY_FIRST = 1,
    INCLUDE_FIRST,
    BEFORE_FIRST,
};

static void vmsetm(void *vd, void *vs2, int len, enum set_mask_type type)
{
    Bool first_mask_bit = False;

    for (int i = 0; i < len; ++i) {
        /* write a zero to all following active elements */
        if (first_mask_bit) {
            vext_set_elem_mask(vd, i, 0);
            continue;
        }
        if (vext_elem_mask(vs2, i)) {
            first_mask_bit = True;
            if (type == BEFORE_FIRST) {
                vext_set_elem_mask(vd, i, 0);
            } else {
                vext_set_elem_mask(vd, i, 1);
            }
        } else {
            if (type == ONLY_FIRST) {
                vext_set_elem_mask(vd, i, 0);
            } else {
                vext_set_elem_mask(vd, i, 1);
            }
        }
    }
}

static void h_Iop_VMsbf_m(void *vd, void *vs2, int len)
{
    vmsetm(vd, vs2, len, BEFORE_FIRST);
}

static void h_Iop_VMsif_m(void *vd, void *vs2, int len)
{
    vmsetm(vd, vs2, len, INCLUDE_FIRST);
}

static void h_Iop_VMsof_m(void *vd, void *vs2, int len)
{
    vmsetm(vd, vs2, len, ONLY_FIRST);
}

/* Vector Iota Instruction */
#define GEN_VEXT_VIOTA_M(NAME, ETYPE)                    \
static void h_Iop_##NAME(void *vd, void *vs2, int len)   \
{                                                        \
    uint32_t sum = 0;                                    \
                                                         \
    for (int i = 0; i < len; ++i) {                      \
        *((ETYPE *)vd + i) = sum;                        \
        if (vext_elem_mask(vs2, i)) {                    \
            sum++;                                       \
        }                                                \
    }                                                    \
}

GEN_VEXT_VIOTA_M(VIota_m_8, uint8_t)
GEN_VEXT_VIOTA_M(VIota_m_16, uint16_t)
GEN_VEXT_VIOTA_M(VIota_m_32, uint32_t)
GEN_VEXT_VIOTA_M(VIota_m_64, uint64_t)

/* Vector Element Index Instruction */
#define GEN_VEXT_VID_V(NAME, ETYPE)                      \
static void h_Iop_##NAME(void *vd, Long dummy, int len)  \
{                                                        \
                                                         \
    for (int i = 0; i < len; ++i) {                      \
        *((ETYPE *)vd + i) = i;                          \
    }                                                    \
}

GEN_VEXT_VID_V(VId_v_8, uint8_t)
GEN_VEXT_VID_V(VId_v_16, uint16_t)
GEN_VEXT_VID_V(VId_v_32, uint32_t)
GEN_VEXT_VID_V(VId_v_64, uint64_t)

/* Vector Compress Instruction */
#define GEN_VEXT_VCOMPRESS_VM(NAME, ETYPE)                        \
static void h_Iop_##NAME(void *vd, void *vs1, void *vs2, int len) \
{                                                                 \
    uint32_t num = 0;                                             \
                                                                  \
    for (int i = 0; i < len; ++i) {                               \
        if (!vext_elem_mask(vs1, i)) {                            \
            continue;                                             \
        }                                                         \
        *((ETYPE *)vd + num) = *((ETYPE *)vs2 + i);               \
        num++;                                                    \
    }                                                             \
}

/* Compress into vd elements of vs2 where vs1 is enabled */
GEN_VEXT_VCOMPRESS_VM(VCompress_vm_8, uint8_t)
GEN_VEXT_VCOMPRESS_VM(VCompress_vm_16, uint16_t)
GEN_VEXT_VCOMPRESS_VM(VCompress_vm_32, uint32_t)
GEN_VEXT_VCOMPRESS_VM(VCompress_vm_64, uint64_t)

/*
 * Vector Permutation Instructions
 */

/* Vector Slide Instructions */
#define GEN_VEXT_VSLIDEUP_VX(NAME, ETYPE)                         \
static void h_Iop_##NAME(void *vd, ULong s1, void *vs2, int len)  \
{                                                                 \
    int offset = s1;                                              \
                                                                  \
    for (int i = offset; i < len; ++i) {                          \
        *((ETYPE *)vd + i) = *((ETYPE *)vs2 + (i - offset));      \
    }                                                             \
}

/* vslideup.vx vd, vs2, rs1, vm # vd[i+rs1] = vs2[i] */
GEN_VEXT_VSLIDEUP_VX(VSlideup_vx_8, uint8_t)
GEN_VEXT_VSLIDEUP_VX(VSlideup_vx_16, uint16_t)
GEN_VEXT_VSLIDEUP_VX(VSlideup_vx_32, uint32_t)
GEN_VEXT_VSLIDEUP_VX(VSlideup_vx_64, uint64_t)

#define GEN_VEXT_VSLIDEDOWN_VX(NAME, ETYPE)                       \
static void h_Iop_##NAME(void *vd, ULong s1, void *vs2, int len,  \
                         int vlmax)                               \
{                                                                 \
    int i_max, i;                                                 \
                                                                  \
    i_max = MIN(s1 < vlmax ? vlmax - s1 : 0, len);                \
    for (i = 0; i < i_max; ++i) {                                 \
        *((ETYPE *)vd + i) = *((ETYPE *)vs2 + (i + s1));          \
    }                                                             \
                                                                  \
    for (i = i_max; i < len; ++i) {                               \
         *((ETYPE *)vd + i) = 0;                                  \
    }                                                             \
}

/* vslidedown.vx vd, vs2, rs1, vm # vd[i] = vs2[i+rs1] */
GEN_VEXT_VSLIDEDOWN_VX(VSlidedown_vx_8, uint8_t)
GEN_VEXT_VSLIDEDOWN_VX(VSlidedown_vx_16, uint16_t)
GEN_VEXT_VSLIDEDOWN_VX(VSlidedown_vx_32, uint32_t)
GEN_VEXT_VSLIDEDOWN_VX(VSlidedown_vx_64, uint64_t)

#define GEN_VEXT_VSLIE1UP(BITWIDTH)                               \
static void vslide1up_##BITWIDTH(void *vd, uint64_t s1,           \
                                 void *vs2, int len)              \
{                                                                 \
    typedef uint##BITWIDTH##_t ETYPE;                             \
                                                                  \
    for (int i = 0; i < len; ++i) {                               \
        if (i == 0) {                                             \
            *((ETYPE *)vd + i) = s1;                              \
        } else {                                                  \
            *((ETYPE *)vd + i) = *((ETYPE *)vs2 + (i - 1));       \
        }                                                         \
    }                                                             \
}

GEN_VEXT_VSLIE1UP(8)
GEN_VEXT_VSLIE1UP(16)
GEN_VEXT_VSLIE1UP(32)
GEN_VEXT_VSLIE1UP(64)

#define GEN_VEXT_VSLIDE1UP_VX(NAME, BITWIDTH)                     \
static void h_Iop_##NAME(void *vd, ULong s1, void *vs2, int len)  \
{                                                                 \
    vslide1up_##BITWIDTH(vd, s1, vs2, len);                       \
}

/* vslide1up.vx vd, vs2, rs1, vm # vd[0]=x[rs1], vd[i+1] = vs2[i] */
GEN_VEXT_VSLIDE1UP_VX(VSlide1up_vx_8, 8)
GEN_VEXT_VSLIDE1UP_VX(VSlide1up_vx_16, 16)
GEN_VEXT_VSLIDE1UP_VX(VSlide1up_vx_32, 32)
GEN_VEXT_VSLIDE1UP_VX(VSlide1up_vx_64, 64)

#define GEN_VEXT_VSLIDE1DOWN(BITWIDTH)                            \
static void vslide1down_##BITWIDTH(void *vd, uint64_t s1,         \
                                   void *vs2, int len)            \
{                                                                 \
    typedef uint##BITWIDTH##_t ETYPE;                             \
                                                                  \
    for (int i = 0; i < len; ++i) {                               \
        if (i == len - 1) {                                       \
            *((ETYPE *)vd + i) = s1;                              \
        } else {                                                  \
            *((ETYPE *)vd + i) = *((ETYPE *)vs2 + (i + 1));       \
        }                                                         \
    }                                                             \
}

GEN_VEXT_VSLIDE1DOWN(8)
GEN_VEXT_VSLIDE1DOWN(16)
GEN_VEXT_VSLIDE1DOWN(32)
GEN_VEXT_VSLIDE1DOWN(64)

#define GEN_VEXT_VSLIDE1DOWN_VX(NAME, BITWIDTH)                   \
static void h_Iop_##NAME(void *vd, ULong s1, void *vs2, int len)  \
{                                                                 \
    vslide1down_##BITWIDTH(vd, s1, vs2, len);                     \
}

/* vslide1down.vx vd, vs2, rs1, vm # vd[i] = vs2[i+1], vd[vl-1]=x[rs1] */
GEN_VEXT_VSLIDE1DOWN_VX(VSlide1down_vx_8, 8)
GEN_VEXT_VSLIDE1DOWN_VX(VSlide1down_vx_16, 16)
GEN_VEXT_VSLIDE1DOWN_VX(VSlide1down_vx_32, 32)
GEN_VEXT_VSLIDE1DOWN_VX(VSlide1down_vx_64, 64)

/* Vector Register Gather Instruction */
#define GEN_VEXT_VRGATHER_VV(NAME, TS1, TS2)                   \
static void h_Iop_##NAME(void *vd, void *vs1, void *vs2,       \
                         int len, int vlmax)                   \
{                                                              \
    uint64_t index;                                            \
                                                               \
    for (int i = 0; i < len; ++i) {                            \
        index = *((TS1 *)vs1 + i);                             \
        if (index >= vlmax) {                                  \
            *((TS2 *)vd + i) = 0;                              \
        } else {                                               \
            *((TS2 *)vd + i) = *((TS2 *)vs2 + index);          \
        }                                                      \
    }                                                          \
}

/* vd[i] = (vs1[i] >= VLMAX) ? 0 : vs2[vs1[i]]; */
GEN_VEXT_VRGATHER_VV(VRgather_vv_8, uint8_t,  uint8_t)
GEN_VEXT_VRGATHER_VV(VRgather_vv_16, uint16_t, uint16_t)
GEN_VEXT_VRGATHER_VV(VRgather_vv_32, uint32_t, uint32_t)
GEN_VEXT_VRGATHER_VV(VRgather_vv_64, uint64_t, uint64_t)

GEN_VEXT_VRGATHER_VV(VRgatherei16_vv_8, uint16_t, uint8_t)
GEN_VEXT_VRGATHER_VV(VRgatherei16_vv_16, uint16_t, uint16_t)
GEN_VEXT_VRGATHER_VV(VRgatherei16_vv_32, uint16_t, uint32_t)
GEN_VEXT_VRGATHER_VV(VRgatherei16_vv_64, uint16_t, uint64_t)

#define GEN_VEXT_VRGATHER_VX(NAME, ETYPE)                         \
static void h_Iop_##NAME(void *vd, void *v0, ULong s1, void *vs2, \
                         int len, int vlmax)                      \
{                                                                 \
    uint64_t index = s1;                                          \
                                                                  \
    for (int i = 0; i < len; ++i) {                               \
        if (index >= vlmax) {                                     \
            *((ETYPE *)vd + i) = 0;                               \
        } else {                                                  \
            *((ETYPE *)vd + i) = *((ETYPE *)vs2 + index);         \
        }                                                         \
    }                                                             \
}

/* vd[i] = (x[rs1] >= VLMAX) ? 0 : vs2[rs1] */
GEN_VEXT_VRGATHER_VX(VRgather_vx_8, uint8_t)
GEN_VEXT_VRGATHER_VX(VRgather_vx_16, uint16_t)
GEN_VEXT_VRGATHER_VX(VRgather_vx_32, uint32_t)
GEN_VEXT_VRGATHER_VX(VRgather_vx_64, uint64_t)

struct Iop_handler {
   const char* name;
   const void* fn;
};

#define H_V_BASIC(op) \
   [Iop_V##op] = {"Iop_V" #op, h_Iop_V##op}

#define H_V1(op) \
   [Iop_V##op##_8]  = {"Iop_V" #op "_8", h_Iop_V##op##_8},   \
   [Iop_V##op##_16] = {"Iop_V" #op "_16", h_Iop_V##op##_16}, \
   [Iop_V##op##_32] = {"Iop_V" #op "_32", h_Iop_V##op##_32}, \
   [Iop_V##op##_64] = {"Iop_V" #op "_64", h_Iop_V##op##_64}

#define H_V1_ALTER(op, alter_op) \
   [Iop_V##op##_8]  = {"Iop_V" #op "_8", h_Iop_V##alter_op##_8},   \
   [Iop_V##op##_16] = {"Iop_V" #op "_16", h_Iop_V##alter_op##_16}, \
   [Iop_V##op##_32] = {"Iop_V" #op "_32", h_Iop_V##alter_op##_32}, \
   [Iop_V##op##_64] = {"Iop_V" #op "_64", h_Iop_V##alter_op##_64}

#define H_V1_IEXT(op) \
   [Iop_V##op##_vf2_16] = {"Iop_V" #op "_vf2_16", h_Iop_V##op##_vf2_16}, \
   [Iop_V##op##_vf2_32] = {"Iop_V" #op "_vf2_32", h_Iop_V##op##_vf2_32}, \
   [Iop_V##op##_vf2_64] = {"Iop_V" #op "_vf2_64", h_Iop_V##op##_vf2_64}, \
   [Iop_V##op##_vf4_32] = {"Iop_V" #op "_vf4_32", h_Iop_V##op##_vf4_32}, \
   [Iop_V##op##_vf4_64] = {"Iop_V" #op "_vf4_64", h_Iop_V##op##_vf4_64}, \
   [Iop_V##op##_vf8_64] = {"Iop_V" #op "_vf8_64", h_Iop_V##op##_vf8_64}

#define H_V_V(op) \
   [Iop_V##op##_vv_8]  = {"Iop_V" #op "_vv_8", h_Iop_V##op##_vv_8},   \
   [Iop_V##op##_vv_16] = {"Iop_V" #op "_vv_16", h_Iop_V##op##_vv_16}, \
   [Iop_V##op##_vv_32] = {"Iop_V" #op "_vv_32", h_Iop_V##op##_vv_32}, \
   [Iop_V##op##_vv_64] = {"Iop_V" #op "_vv_64", h_Iop_V##op##_vv_64}

#define H_V_X(op) \
   [Iop_V##op##_vx_8]  = {"Iop_V" #op "_vx_8", h_Iop_V##op##_vx_8},   \
   [Iop_V##op##_vx_16] = {"Iop_V" #op "_vx_16", h_Iop_V##op##_vx_16}, \
   [Iop_V##op##_vx_32] = {"Iop_V" #op "_vx_32", h_Iop_V##op##_vx_32}, \
   [Iop_V##op##_vx_64] = {"Iop_V" #op "_vx_64", h_Iop_V##op##_vx_64}

#define H_V_I(op) \
   [Iop_V##op##_vi_8]  = {"Iop_V" #op "_vi_8", h_Iop_V##op##_vx_8},   \
   [Iop_V##op##_vi_16] = {"Iop_V" #op "_vi_16", h_Iop_V##op##_vx_16}, \
   [Iop_V##op##_vi_32] = {"Iop_V" #op "_vi_32", h_Iop_V##op##_vx_32}, \
   [Iop_V##op##_vi_64] = {"Iop_V" #op "_vi_64", h_Iop_V##op##_vx_64}

#define H_V_M(op) \
   [Iop_V##op##_vm_8]  = {"Iop_V" #op "_vm_8", h_Iop_V##op##_vm_8},   \
   [Iop_V##op##_vm_16] = {"Iop_V" #op "_vm_16", h_Iop_V##op##_vm_16}, \
   [Iop_V##op##_vm_32] = {"Iop_V" #op "_vm_32", h_Iop_V##op##_vm_32}, \
   [Iop_V##op##_vm_64] = {"Iop_V" #op "_vm_64", h_Iop_V##op##_vm_64}

#define H_V_VX(op) \
   H_V_V(op), \
   H_V_X(op)

#define H_V_VI(op) \
   H_V_V(op), \
   H_V_I(op)

#define H_V_XI(op) \
   H_V_X(op), \
   H_V_I(op)

#define H_V_VXI(op) \
   H_V_V(op), \
   H_V_X(op), \
   H_V_I(op)

#define H_W_V(op) \
   [Iop_V##op##_wv_8]  = {"Iop_V" #op "_wv_8", h_Iop_V##op##_wv_8},   \
   [Iop_V##op##_wv_16] = {"Iop_V" #op "_wv_16", h_Iop_V##op##_wv_16}, \
   [Iop_V##op##_wv_32] = {"Iop_V" #op "_wv_32", h_Iop_V##op##_wv_32}

#define H_W_X(op) \
   [Iop_V##op##_wx_8]  = {"Iop_V" #op "_wx_8", h_Iop_V##op##_wx_8},   \
   [Iop_V##op##_wx_16] = {"Iop_V" #op "_wx_16", h_Iop_V##op##_wx_16}, \
   [Iop_V##op##_wx_32] = {"Iop_V" #op "_wx_32", h_Iop_V##op##_wx_32}

#define H_W_I(op) \
   [Iop_V##op##_wi_8]  = {"Iop_V" #op "_wi_8", h_Iop_V##op##_wx_8},   \
   [Iop_V##op##_wi_16] = {"Iop_V" #op "_wi_16", h_Iop_V##op##_wx_16}, \
   [Iop_V##op##_wi_32] = {"Iop_V" #op "_wi_32", h_Iop_V##op##_wx_32}

#define H_W_VXI(op) \
   H_W_V(op), \
   H_W_X(op), \
   H_W_I(op)

#define HW_V_V(op) \
   [Iop_V##op##_vv_8]  = {"Iop_V" #op "_vv_8", h_Iop_V##op##_vv_8},   \
   [Iop_V##op##_vv_16] = {"Iop_V" #op "_vv_16", h_Iop_V##op##_vv_16}, \
   [Iop_V##op##_vv_32] = {"Iop_V" #op "_vv_32", h_Iop_V##op##_vv_32}

#define HW_V_X(op) \
   [Iop_V##op##_vx_8]  = {"Iop_V" #op "_vx_8", h_Iop_V##op##_vx_8},   \
   [Iop_V##op##_vx_16] = {"Iop_V" #op "_vx_16", h_Iop_V##op##_vx_16}, \
   [Iop_V##op##_vx_32] = {"Iop_V" #op "_vx_32", h_Iop_V##op##_vx_32}

#define HW_W_V(op) \
   [Iop_V##op##_wv_8]  = {"Iop_V" #op "_wv_8", h_Iop_V##op##_wv_8},   \
   [Iop_V##op##_wv_16] = {"Iop_V" #op "_wv_16", h_Iop_V##op##_wv_16}, \
   [Iop_V##op##_wv_32] = {"Iop_V" #op "_wv_32", h_Iop_V##op##_wv_32}

#define HW_W_X(op) \
   [Iop_V##op##_wx_8]  = {"Iop_V" #op "_wx_8", h_Iop_V##op##_wx_8},   \
   [Iop_V##op##_wx_16] = {"Iop_V" #op "_wx_16", h_Iop_V##op##_wx_16}, \
   [Iop_V##op##_wx_32] = {"Iop_V" #op "_wx_32", h_Iop_V##op##_wx_32}

#define HW_V_VX(op) \
   HW_V_V(op), \
   HW_V_X(op)

#define HW_W_VX(op) \
   HW_W_V(op), \
   HW_W_X(op)

#define HW_VW_VX(op) \
   HW_V_VX(op), \
   HW_W_VX(op)

#define H_V_V_M(op) \
   [Iop_V##op##_vvm_8]  = {"Iop_V" #op "_vvm_8", h_Iop_V##op##_vvm_8},   \
   [Iop_V##op##_vvm_16] = {"Iop_V" #op "_vvm_16", h_Iop_V##op##_vvm_16}, \
   [Iop_V##op##_vvm_32] = {"Iop_V" #op "_vvm_32", h_Iop_V##op##_vvm_32}, \
   [Iop_V##op##_vvm_64] = {"Iop_V" #op "_vvm_64", h_Iop_V##op##_vvm_64}

#define H_V_X_M(op) \
   [Iop_V##op##_vxm_8]  = {"Iop_V" #op "_vxm_8", h_Iop_V##op##_vxm_8},   \
   [Iop_V##op##_vxm_16] = {"Iop_V" #op "_vxm_16", h_Iop_V##op##_vxm_16}, \
   [Iop_V##op##_vxm_32] = {"Iop_V" #op "_vxm_32", h_Iop_V##op##_vxm_32}, \
   [Iop_V##op##_vxm_64] = {"Iop_V" #op "_vxm_64", h_Iop_V##op##_vxm_64}

#define H_V_I_M(op) \
   [Iop_V##op##_vim_8]  = {"Iop_V" #op "_vim_8", h_Iop_V##op##_vxm_8},   \
   [Iop_V##op##_vim_16] = {"Iop_V" #op "_vim_16", h_Iop_V##op##_vxm_16}, \
   [Iop_V##op##_vim_32] = {"Iop_V" #op "_vim_32", h_Iop_V##op##_vxm_32}, \
   [Iop_V##op##_vim_64] = {"Iop_V" #op "_vim_64", h_Iop_V##op##_vxm_64}

#define H_V_VX_M(op) \
   H_V_V_M(op), \
   H_V_X_M(op)

#define H_V_VXI_M(op) \
   H_V_V_M(op), \
   H_V_X_M(op), \
   H_V_I_M(op)

#define H_RED(op) \
   [Iop_V##op##_vs_8]  = {"Iop_V" #op "_vs_8", h_Iop_V##op##_vs_8},         \
   [Iop_V##op##_vs_16] = {"Iop_V" #op "_vs_16", h_Iop_V##op##_vs_16},       \
   [Iop_V##op##_vs_32] = {"Iop_V" #op "_vs_32", h_Iop_V##op##_vs_32},       \
   [Iop_V##op##_vs_64] = {"Iop_V" #op "_vs_64", h_Iop_V##op##_vs_64},       \
   [Iop_V##op##_vsm_8]  = {"Iop_V" #op "_vsm_8", h_Iop_V##op##_vsm_8},   \
   [Iop_V##op##_vsm_16] = {"Iop_V" #op "_vsm_16", h_Iop_V##op##_vsm_16}, \
   [Iop_V##op##_vsm_32] = {"Iop_V" #op "_vsm_32", h_Iop_V##op##_vsm_32}, \
   [Iop_V##op##_vsm_64] = {"Iop_V" #op "_vsm_64", h_Iop_V##op##_vsm_64}

#define HW_RED(op) \
   [Iop_V##op##_vs_8]  = {"Iop_V" #op "_vs_8", h_Iop_V##op##_vs_8},         \
   [Iop_V##op##_vs_16] = {"Iop_V" #op "_vs_16", h_Iop_V##op##_vs_16},       \
   [Iop_V##op##_vs_32] = {"Iop_V" #op "_vs_32", h_Iop_V##op##_vs_32},       \
   [Iop_V##op##_vsm_8]  = {"Iop_V" #op "_vsm_8", h_Iop_V##op##_vsm_8},   \
   [Iop_V##op##_vsm_16] = {"Iop_V" #op "_vsm_16", h_Iop_V##op##_vsm_16}, \
   [Iop_V##op##_vsm_32] = {"Iop_V" #op "_vsm_32", h_Iop_V##op##_vsm_32}

#define H_M_M(op) \
   [Iop_V##op##_mm]  = {"Iop_V" #op "_mm", h_Iop_V##op##_mm}

static const struct Iop_handler IOP_HANDLERS[] = {
   H_V_VXI(And),
   H_V_VXI(Or),
   H_V_VXI(Xor),
   H_V_VXI(Add),

   H_V_VX(Sub),
   H_V_XI(Rsub),

   H_V_VX(Min),
   H_V_VX(Minu),
   H_V_VX(Max),
   H_V_VX(Maxu),

   H_V_VX(Mul),
   H_V_VX(Mulh),
   H_V_VX(Mulhu),
   H_V_VX(Mulhsu),

   H_V_VX(Divu),
   H_V_VX(Div),
   H_V_VX(Remu),
   H_V_VX(Rem),

   H_V_VXI(Mseq),
   H_V_VXI(Msne),
   H_V_VX(Msltu),
   H_V_VX(Mslt),
   H_V_VXI(Msleu),
   H_V_VXI(Msle),
   H_V_XI(Msgtu),
   H_V_XI(Msgt),

   H_V_VX(Macc),
   H_V_VX(Nmsac),
   H_V_VX(Madd),
   H_V_VX(Nmsub),

   H_V_VXI(Sll),
   H_V_VXI(Srl),
   H_V_VXI(Sra),

   H_W_VXI(Nsrl),
   H_W_VXI(Nsra),

   HW_VW_VX(Waddu),
   HW_VW_VX(Wadd),
   HW_VW_VX(Wsubu),
   HW_VW_VX(Wsub),

   HW_V_VX(Wmulu),
   HW_V_VX(Wmulsu),
   HW_V_VX(Wmul),

   HW_V_VX(Wmaccu),
   HW_V_VX(Wmacc),
   HW_V_VX(Wmaccsu),
   HW_V_X(Wmaccus),

   H_V_VXI_M(Merge),

   H_V_VXI_M(Adc),
   H_V_VX_M(Sbc),

   H_V_VXI_M(Madc),
   H_V_VXI(Madc),
   H_V_VX_M(Msbc),
   H_V_VX(Msbc),

   H_RED(Redsum),
   H_RED(Redand),
   H_RED(Redor),
   H_RED(Redxor),
   H_RED(Redminu),
   H_RED(Redmin),
   H_RED(Redmaxu),
   H_RED(Redmax),

   HW_RED(Wredsumu),
   HW_RED(Wredsum),

   H_V1_IEXT(Zext),
   H_V1_IEXT(Sext),

   H_V1(Not),
   H_V1(CmpNEZ),
   H_V1(ExpandBitsTo),

   H_V1(Mv_v_v),
   H_V1(Mv_v_x),
   H_V1_ALTER(Mv_v_i, Mv_v_x),

   H_V1(Id_v),
   H_V1(Iota_m),

   H_M_M(Mand),
   H_M_M(Mnand),
   H_M_M(Mandn),
   H_M_M(Mxor),
   H_M_M(Mor),
   H_M_M(Mnor),
   H_M_M(Morn),
   H_M_M(Mxnor),

   H_V_BASIC(Cpop_m),
   H_V_BASIC(First_m),

   H_V_BASIC(Msbf_m),
   H_V_BASIC(Msif_m),
   H_V_BASIC(Msof_m),

   H_V_XI(Slideup),
   H_V_XI(Slidedown),
   H_V_X(Slide1up),
   H_V_X(Slide1down),

   H_V_VXI(Rgather),
   H_V_V(Rgatherei16),

   H_V_M(Compress),

   [Iop_LAST] = {"Iop_LAST", 0}
};

#define MAX_REGS 8
#define MAX_ARGS_STACK_SIZE 1000

static void adjust_sp(ISelEnv* env, Long n)
{
   if (n >= -2048 && n < 2048) {
      addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, hregRISCV64_x2(),
               hregRISCV64_x2(), n));
   } else {
      HReg tmp = newVRegI(env);
      addInstr(env, RISCV64Instr_LI(tmp, n));
      addInstr(env, RISCV64Instr_ALU(RISCV64op_ADD, hregRISCV64_x2(),
               hregRISCV64_x2(), tmp));
   }
}

static void storeVecReg(ISelEnv* env, HReg src[], Int nregs, HReg addr)
{
   const Int vlen_b = VLEN / 8;
   Int vl_ldst64 = vlen_b / 8;

   HReg tmp = newVRegI(env);
   addInstr(env, RISCV64Instr_MV(tmp, addr));

   for (int i = 0; i < nregs; ++i) {
      addInstr(env, RISCV64Instr_VStore(RISCV64op_VStore64, vl_ldst64, src[i], tmp));
      addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, tmp, tmp, vlen_b));
   }
}

static void loadVecReg(ISelEnv* env, HReg dst[], Int nregs, HReg addr)
{
   const Int vlen_b = VLEN / 8;
   Int vl_ldst64 = vlen_b / 8;

   HReg tmp = newVRegI(env);
   addInstr(env, RISCV64Instr_MV(tmp, addr));

   for (int i = 0; i < nregs; ++i) {
      addInstr(env, RISCV64Instr_VLoad(RISCV64op_VLoad64, vl_ldst64, dst[i], tmp));
      addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, tmp, tmp, vlen_b));
   }
}

static Bool iselVecExpr_R_wrk_unop(HReg dst[], ISelEnv* env, IRExpr* e)
{
   const void* fn = IOP_HANDLERS[e->Iex.Unop.op & IR_OP_MASK].fn;
   if (fn == 0) {
      return False;
   }

   const Int vlen_b = VLEN / 8;
   Int sz = 8;  // address gap between parameters

   Int dst_nregs = 1;
   IRType dst_ty = typeOfIRExpr(env->type_env, e);
   if (isVecIRType(dst_ty)) {
      Int dst_sz = ROUND_UP(sizeofVecIRType(dst_ty), vlen_b);
      dst_nregs = dst_sz / vlen_b;
      sz = MAX(sz, dst_sz);
   }

   Int src_nregs = 0;
   HReg src[MAX_REGS] = {0};
   IRType src_ty = typeOfIRExpr(env->type_env, e->Iex.Unop.arg);
   if (isVecIRType(src_ty)) {
      Int src_sz = sizeofVecIRType(src_ty);
      src_nregs = DIV_ROUND_UP(src_sz, vlen_b);
      sz = MAX(src_sz, sz);

      iselVecExpr_R(src, env, e->Iex.Unop.arg);
   } else {
      src[0] = iselIntExpr_R(env, e->Iex.Unop.arg);
   }

   HReg argp = newVRegI(env);
   adjust_sp(env, -MAX_ARGS_STACK_SIZE);
   addInstr(env, RISCV64Instr_MV(argp, hregRISCV64_x2()));
   addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, argp, argp, 15));
   addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ANDI, argp, argp, ~(Int)15));

   // a0 - dst
   addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, hregRISCV64_x10(), argp, 0));

   // a1 - src
   if (isVecIRType(src_ty)) {
      addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, hregRISCV64_x11(), argp, sz));
      storeVecReg(env, src, src_nregs, hregRISCV64_x11());
   } else {
      addInstr(env, RISCV64Instr_MV(hregRISCV64_x11(), src[0]));
   }

   // a2 - vl
   HReg base = get_baseblock_register();
   Int  off  = offsetof(VexGuestRISCV64State, guest_vl) - BASEBLOCK_OFFSET_ADJUSTMENT;
   vassert(off >= -2048 && off < 2048);
   addInstr(env, RISCV64Instr_Load(RISCV64op_LW, hregRISCV64_x12(), base, off));

   addInstr(env, RISCV64Instr_Call(mk_RetLoc_simple(RLPri_None), (Addr64) fn, INVALID_HREG, 3, 0));
   if (isVecIRType(dst_ty)) {
      loadVecReg(env, dst, dst_nregs, argp);
   } else {
      addInstr(env, RISCV64Instr_Load(RISCV64op_LD, dst[0], argp, 0));
   }

   adjust_sp(env, MAX_ARGS_STACK_SIZE);
   return True;
}

static Bool iselVecExpr_R_wrk_binop(HReg dst[], ISelEnv* env, IRExpr* e, Bool extra)
{
   const void* fn = IOP_HANDLERS[e->Iex.Binop.op & IR_OP_MASK].fn;
   if (fn == 0) {
      return False;
   }

   const Int vlen_b = VLEN / 8;
   Int sz = 0;  // address gap between parameters

   HReg src1[MAX_REGS] = {0};
   HReg src2[MAX_REGS] = {0};

   HReg base = get_baseblock_register();
   Int  off  = 0;

   Int dst_nregs = 1;
   IRType dst_ty = typeOfIRExpr(env->type_env, e);
   if (isVecIRType(dst_ty)) {
      Int dst_sz = ROUND_UP(sizeofVecIRType(dst_ty), vlen_b);
      dst_nregs = dst_sz / vlen_b;
      sz = MAX(sz, dst_sz);
   }

   Int src1_nregs = 0;
   IRType src1_ty = typeOfIRExpr(env->type_env, e->Iex.Binop.arg1);
   if (isVecIRType(src1_ty)) {
      src1_nregs = DIV_ROUND_UP(sizeofVecIRType(src1_ty), vlen_b);

      iselVecExpr_R(src1, env, e->Iex.Binop.arg1);
   } else {
      src1[0] = iselIntExpr_R(env, e->Iex.Binop.arg1);
   }

   Int src2_nregs = 0;
   IRType src2_ty = typeOfIRExpr(env->type_env, e->Iex.Binop.arg2);
   if (isVecIRType(src2_ty)) {
      Int src2_sz = ROUND_UP(sizeofVecIRType(src2_ty), vlen_b);
      src2_nregs = src2_sz / vlen_b;
      sz = MAX(sz, src2_sz);

      iselVecExpr_R(src2, env, e->Iex.Binop.arg2);
   } else {
      src2[0] = iselIntExpr_R(env, e->Iex.Binop.arg2);
   }

   HReg argp = newVRegI(env);
   adjust_sp(env, -MAX_ARGS_STACK_SIZE);
   addInstr(env, RISCV64Instr_MV(argp, hregRISCV64_x2()));
   addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, argp, argp, 15));
   addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ANDI, argp, argp, ~(Int)15));

   // a0 - dst
   addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, hregRISCV64_x10(), argp, 0));

   // a1 - src1
   if (isVecIRType(src1_ty)) {
      addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, hregRISCV64_x11(), argp, sz));
      storeVecReg(env, src1, src1_nregs, hregRISCV64_x11());
   } else {
      addInstr(env, RISCV64Instr_MV(hregRISCV64_x11(), src1[0]));
   }

   // a2 - src2
   if (isVecIRType(src2_ty)) {
      addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, hregRISCV64_x12(), argp, sz * 2));
      storeVecReg(env, src2, src2_nregs, hregRISCV64_x12());
   } else {
      addInstr(env, RISCV64Instr_MV(hregRISCV64_x12(), src2[0]));
   }

   // a3 - vl
   off = offsetof(VexGuestRISCV64State, guest_vl) - BASEBLOCK_OFFSET_ADJUSTMENT;
   vassert(off >= -2048 && off < 2048);
   addInstr(env, RISCV64Instr_Load(RISCV64op_LW, hregRISCV64_x13(), base, off));

   UChar nargs = 4;
   if (extra) {
      ++nargs;
      // a4 - vlmax, so far only vlmax in extra field

      off = offsetof(VexGuestRISCV64State, guest_vlmax) - BASEBLOCK_OFFSET_ADJUSTMENT;
      vassert(off >= -2048 && off < 2048);
      addInstr(env, RISCV64Instr_Load(RISCV64op_LW, hregRISCV64_x14(), base, off));
   }

   addInstr(env, RISCV64Instr_Call(mk_RetLoc_simple(RLPri_None), (Addr64) fn, INVALID_HREG, nargs, 0));
   loadVecReg(env, dst, dst_nregs, argp);

   adjust_sp(env, MAX_ARGS_STACK_SIZE);
   return True;
}

static Bool iselVecExpr_R_wrk_triop_sss(HReg dst[], ISelEnv* env, IRExpr* e)
{
   const void* fn = IOP_HANDLERS[e->Iex.Triop.details->op & IR_OP_MASK].fn;
   if (fn == 0) {
      return False;
   }

   const Int vlen_b = VLEN / 8;
   Int sz = 0;  // address gap between parameters

   HReg src1[MAX_REGS] = {0};
   HReg src2[MAX_REGS] = {0};
   HReg src3[MAX_REGS] = {0};

   HReg base = get_baseblock_register();
   Int  off  = 0;

   Int dst_nregs = 1;
   IRType dst_ty = typeOfIRExpr(env->type_env, e);
   if (isVecIRType(dst_ty)) {
      Int dst_sz = ROUND_UP(sizeofVecIRType(dst_ty), vlen_b);
      dst_nregs = dst_sz / vlen_b;
      sz = MAX(sz, dst_sz);
   }

   Int src1_nregs = 0;
   IRType src1_ty = typeOfIRExpr(env->type_env, e->Iex.Triop.details->arg1);
   if (isVecIRType(src1_ty)) {
      src1_nregs = DIV_ROUND_UP(sizeofVecIRType(src1_ty), vlen_b);

      iselVecExpr_R(src1, env, e->Iex.Triop.details->arg1);
   } else {
      src1[0] = iselIntExpr_R(env, e->Iex.Triop.details->arg1);
   }

   Int src2_nregs = 0;
   IRType src2_ty = typeOfIRExpr(env->type_env, e->Iex.Triop.details->arg2);
   if (isVecIRType(src2_ty)) {
      Int src2_sz = ROUND_UP(sizeofVecIRType(src2_ty), vlen_b);
      src2_nregs = src2_sz / vlen_b;
      sz = MAX(sz, src2_sz);

      iselVecExpr_R(src2, env, e->Iex.Triop.details->arg2);
   } else {
      src2[0] = iselIntExpr_R(env, e->Iex.Triop.details->arg2);
   }

   Int src3_nregs = 0;
   IRType src3_ty = typeOfIRExpr(env->type_env, e->Iex.Triop.details->arg3);
   if (isVecIRType(src3_ty)) {
      src3_nregs = DIV_ROUND_UP(sizeofVecIRType(src3_ty), vlen_b);

      iselVecExpr_R(src3, env, e->Iex.Triop.details->arg3);
   } else {
      src3[0] = iselIntExpr_R(env, e->Iex.Triop.details->arg3);
   }

   HReg argp = newVRegI(env);
   adjust_sp(env, -MAX_ARGS_STACK_SIZE);
   addInstr(env, RISCV64Instr_MV(argp, hregRISCV64_x2()));
   addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, argp, argp, 15));
   addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ANDI, argp, argp, ~(Int)15));

   // a0 - dst
   addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, hregRISCV64_x10(), argp, 0));

   // a1 - src1
   if (isVecIRType(src1_ty)) {
      addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, hregRISCV64_x11(), argp, sz));
      storeVecReg(env, src1, src1_nregs, hregRISCV64_x11());
   } else {
      addInstr(env, RISCV64Instr_MV(hregRISCV64_x11(), src1[0]));
   }


   // a2 - src2
   if (isVecIRType(src2_ty)) {
      addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, hregRISCV64_x12(), argp, sz * 2));
      storeVecReg(env, src2, src2_nregs, hregRISCV64_x12());
   } else {
      addInstr(env, RISCV64Instr_MV(hregRISCV64_x12(), src2[0]));
   }

   // a3 - src3
   if (isVecIRType(src3_ty)) {
      addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, hregRISCV64_x13(), argp, sz * 3));
      storeVecReg(env, src3, src3_nregs, hregRISCV64_x13());
   } else {
      addInstr(env, RISCV64Instr_MV(hregRISCV64_x13(), src3[0]));
   }

   // a4 - vl
   off = offsetof(VexGuestRISCV64State, guest_vl) - BASEBLOCK_OFFSET_ADJUSTMENT;
   vassert(off >= -2048 && off < 2048);
   addInstr(env, RISCV64Instr_Load(RISCV64op_LW, hregRISCV64_x14(), base, off));

   addInstr(env, RISCV64Instr_Call(mk_RetLoc_simple(RLPri_None), (Addr64) fn, INVALID_HREG, 5, 0));
   loadVecReg(env, dst, dst_nregs, argp);

   adjust_sp(env, MAX_ARGS_STACK_SIZE);
   return True;
}

static Bool iselVecExpr_R_wrk_triop_ssd(HReg dst[], ISelEnv* env, IRExpr* e)
{
   const void* fn = IOP_HANDLERS[e->Iex.Triop.details->op & IR_OP_MASK].fn;
   if (fn == 0) {
      return False;
   }

   const Int vlen_b = VLEN / 8;
   Int sz = 0;  // address gap between parameters

   HReg src1[MAX_REGS] = {0};
   HReg src2[MAX_REGS] = {0};

   HReg base = get_baseblock_register();
   Int  off  = 0;

   Int dst_nregs = 1;
   IRType dst_ty = typeOfIRExpr(env->type_env, e);
   if (isVecIRType(dst_ty)) {
      Int dst_sz = ROUND_UP(sizeofVecIRType(dst_ty), vlen_b);
      dst_nregs = dst_sz / vlen_b;
      sz = MAX(sz, dst_sz);
   }

   Int src1_nregs = 0;
   IRType src1_ty = typeOfIRExpr(env->type_env, e->Iex.Triop.details->arg1);
   if (isVecIRType(src1_ty)) {
      src1_nregs = DIV_ROUND_UP(sizeofVecIRType(src1_ty), vlen_b);

      iselVecExpr_R(src1, env, e->Iex.Triop.details->arg1);
   } else {
      src1[0] = iselIntExpr_R(env, e->Iex.Triop.details->arg1);
   }

   Int src2_nregs = 0;
   IRType src2_ty = typeOfIRExpr(env->type_env, e->Iex.Triop.details->arg2);
   if (isVecIRType(src2_ty)) {
      Int src2_sz = ROUND_UP(sizeofVecIRType(src2_ty), vlen_b);
      src2_nregs = src2_sz / vlen_b;
      sz = MAX(sz, src2_sz);

      iselVecExpr_R(src2, env, e->Iex.Triop.details->arg2);
   } else {
      src2[0] = iselIntExpr_R(env, e->Iex.Triop.details->arg2);
   }

   IRType src3_ty = typeOfIRExpr(env->type_env, e->Iex.Triop.details->arg3);
   vassert(isVecIRType(src3_ty));
   iselVecExpr_R(dst, env, e->Iex.Triop.details->arg3);

   HReg argp = newVRegI(env);
   adjust_sp(env, -MAX_ARGS_STACK_SIZE);
   addInstr(env, RISCV64Instr_MV(argp, hregRISCV64_x2()));
   addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, argp, argp, 15));
   addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ANDI, argp, argp, ~(Int)15));

   // a0 - dst
   addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, hregRISCV64_x10(), argp, 0));
   storeVecReg(env, dst, dst_nregs, hregRISCV64_x10());

   // a1 - src1
   if (isVecIRType(src1_ty)) {
      addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, hregRISCV64_x11(), argp, sz));
      storeVecReg(env, src1, src1_nregs, hregRISCV64_x11());
   } else {
      addInstr(env, RISCV64Instr_MV(hregRISCV64_x11(), src1[0]));
   }


   // a2 - src2
   if (isVecIRType(src2_ty)) {
      addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, hregRISCV64_x12(), argp, sz * 2));
      storeVecReg(env, src2, src2_nregs, hregRISCV64_x12());
   } else {
      addInstr(env, RISCV64Instr_MV(hregRISCV64_x12(), src2[0]));
   }

   // a3 - vl
   off = offsetof(VexGuestRISCV64State, guest_vl) - BASEBLOCK_OFFSET_ADJUSTMENT;
   vassert(off >= -2048 && off < 2048);
   addInstr(env, RISCV64Instr_Load(RISCV64op_LW, hregRISCV64_x13(), base, off));

   addInstr(env, RISCV64Instr_Call(mk_RetLoc_simple(RLPri_None), (Addr64) fn, INVALID_HREG, 4, 0));
   loadVecReg(env, dst, dst_nregs, argp);

   adjust_sp(env, MAX_ARGS_STACK_SIZE);
   return True;
}

static void iselVecExpr_R_wrk(HReg dst[], ISelEnv* env, IRExpr* e)
{
   Int vlen_b = VLEN / 8;
   Int vl_ldst64 = vlen_b / 8;
   IRType ty = typeOfIRExpr(env->type_env, e);
   Int nregs = 0;
   if (isVecIRType(ty)) {
      nregs = DIV_ROUND_UP(sizeofVecIRType(ty), vlen_b);
      for (int i = 0; i < nregs; ++i) {
         dst[i] = newVRegV(env);
      }
   } else {
      dst[0] = newVRegI(env);
   }

   switch (e->tag) {
      case Iex_RdTmp: {
         lookupIRTempVec(dst, nregs, env, e->Iex.RdTmp.tmp);
         return;
      }
      case Iex_Get: {
         HReg base = newVRegI(env);
         Int  off  = e->Iex.Get.offset - BASEBLOCK_OFFSET_ADJUSTMENT;
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, base, get_baseblock_register(), off));
         addInstr(env, RISCV64Instr_VLoad(RISCV64op_VLoad64, vl_ldst64, dst[0], base));

         for (int i = 1; i < nregs; ++i) {
            addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, base, base, vlen_b));
            addInstr(env, RISCV64Instr_VLoad(RISCV64op_VLoad64, vl_ldst64, dst[i], base));
         }
         return;
      }

      case Iex_Unop: {
         Bool ret = iselVecExpr_R_wrk_unop(dst, env, e);
         if (ret == True) {
            return;
         }

         /*
         RISCV64VALUOp op;
         IROp ir_op = e->Iex.Unop.op & IR_OP_MASK;
         switch (ir_op) {
            default:
               goto irreducible;
         }

         dst  = newVRegV(env);
         HReg argL = iselVecExpr_R(env, e->Iex.Unop.arg);
         HReg argR = hregRISCV64_x0();
         addInstr(env, RISCV64Instr_VALU(op, vl, dst, argL, argR));
         return dst;
         */
         break;
      }
      case Iex_Binop: {
         Bool extra = False;
         switch (e->Iex.Binop.op & IR_OP_MASK) {
            case Iop_VSlidedown_vx_8 ... Iop_VSlidedown_vx_64:
            case Iop_VSlidedown_vi_8 ... Iop_VSlidedown_vi_64:
            case Iop_VRgather_vv_8 ... Iop_VRgather_vv_64:
            case Iop_VRgather_vx_8 ... Iop_VRgather_vx_64:
            case Iop_VRgather_vi_8 ... Iop_VRgather_vi_64:
            case Iop_VRgatherei16_vv_8 ... Iop_VRgatherei16_vv_64:
               extra = True;
               break;
            default:
               break;
         }

         Bool ret = iselVecExpr_R_wrk_binop(dst, env, e, extra);
         if (ret == True) {
            return;
         }

         /*
         RISCV64VALUOp op;
         IROp ir_op = e->Iex.Binop.op & IR_OP_MASK;
         switch (ir_op) {
            case Iop_VAdd8 ... Iop_VAdd64:
               op = RISCV64op_VADD8 + (ir_op - Iop_VAdd8); break;
            case Iop_VOr8 ... Iop_VOr64:
               op = RISCV64op_VOr8 + (ir_op - Iop_VOr8); break;
            default:
               goto irreducible;
         }

         dst  = newVRegV(env);
         HReg argL = iselVecExpr_R(env, e->Iex.Binop.arg1);
         HReg argR = iselVecExpr_R(env, e->Iex.Binop.arg2);
         addInstr(env, RISCV64Instr_VALU(op, vl, dst, argL, argR));
         return dst;
         */
         break;
      }
      case Iex_Triop: {
         Bool ret = False;
         IROp bop = e->Iex.Triop.details->op & IR_OP_MASK;
         if (bop > Iop_SSS_Start && bop < Iop_SSS_End) {
            ret = iselVecExpr_R_wrk_triop_sss(dst, env, e);
         } else if (bop > Iop_SSD_Start && bop < Iop_SSD_End) {
            ret = iselVecExpr_R_wrk_triop_ssd(dst, env, e);
         }

         if (ret == True) {
            return;
         }
         break;
      }

      default:
         goto irreducible;
   }

irreducible:
   ppIRExpr(e);
   vpanic("iselVecExpr_R(riscv64)");
}

static void iselVecExpr_R(HReg r[], ISelEnv* env, IRExpr* e)
{
   iselVecExpr_R_wrk(r, env, e);

   /* Sanity checks ... */
   //vassert(hregClass(r) == HRcVecVLen);
   //vassert(hregIsVirtual(r));
}

/*------------------------------------------------------------*/
/*--- ISEL: Floating point expressions                     ---*/
/*------------------------------------------------------------*/

/* DO NOT CALL THIS DIRECTLY ! */
static HReg iselFltExpr_wrk(ISelEnv* env, IRExpr* e)
{
   IRType ty = typeOfIRExpr(env->type_env, e);
   vassert(ty == Ity_F32 || ty == Ity_F64);

   switch (e->tag) {
   /* ------------------------ TEMP ------------------------- */
   case Iex_RdTmp: {
      return lookupIRTemp(env, e->Iex.RdTmp.tmp);
   }

   /* ------------------------ LOAD ------------------------- */
   case Iex_Load: {
      if (e->Iex.Load.end != Iend_LE)
         goto irreducible;

      HReg dst = newVRegF(env);
      /* TODO Optimize the cases with small imm Add64/Sub64. */
      HReg addr = iselIntExpr_R(env, e->Iex.Load.addr);

      if (ty == Ity_F32)
         addInstr(env, RISCV64Instr_FpLdSt(RISCV64op_FLW, dst, addr, 0));
      else if (ty == Ity_F64)
         addInstr(env, RISCV64Instr_FpLdSt(RISCV64op_FLD, dst, addr, 0));
      else
         vassert(0);
      return dst;
   }

   /* -------------------- QUATERNARY OP -------------------- */
   case Iex_Qop: {
      switch (e->Iex.Qop.details->op) {
      case Iop_MAddF32: {
         HReg dst  = newVRegF(env);
         HReg argN = iselFltExpr(env, e->Iex.Qop.details->arg2);
         HReg argM = iselFltExpr(env, e->Iex.Qop.details->arg3);
         HReg argA = iselFltExpr(env, e->Iex.Qop.details->arg4);
         set_fcsr_rounding_mode(env, e->Iex.Qop.details->arg1);
         addInstr(env, RISCV64Instr_FpTernary(RISCV64op_FMADD_S, dst, argN,
                                              argM, argA));
         return dst;
      }
      case Iop_MAddF64: {
         HReg dst  = newVRegF(env);
         HReg argN = iselFltExpr(env, e->Iex.Qop.details->arg2);
         HReg argM = iselFltExpr(env, e->Iex.Qop.details->arg3);
         HReg argA = iselFltExpr(env, e->Iex.Qop.details->arg4);
         set_fcsr_rounding_mode(env, e->Iex.Qop.details->arg1);
         addInstr(env, RISCV64Instr_FpTernary(RISCV64op_FMADD_D, dst, argN,
                                              argM, argA));
         return dst;
      }
      default:
         break;
      }

      break;
   }

   /* --------------------- TERNARY OP ---------------------- */
   case Iex_Triop: {
      RISCV64FpBinaryOp op;
      switch (e->Iex.Triop.details->op) {
      case Iop_AddF32:
         op = RISCV64op_FADD_S;
         break;
      case Iop_MulF32:
         op = RISCV64op_FMUL_S;
         break;
      case Iop_DivF32:
         op = RISCV64op_FDIV_S;
         break;
      case Iop_AddF64:
         op = RISCV64op_FADD_D;
         break;
      case Iop_SubF64:
         op = RISCV64op_FSUB_D;
         break;
      case Iop_MulF64:
         op = RISCV64op_FMUL_D;
         break;
      case Iop_DivF64:
         op = RISCV64op_FDIV_D;
         break;
      default:
         goto irreducible;
      }
      HReg dst  = newVRegF(env);
      HReg src1 = iselFltExpr(env, e->Iex.Triop.details->arg2);
      HReg src2 = iselFltExpr(env, e->Iex.Triop.details->arg3);
      set_fcsr_rounding_mode(env, e->Iex.Triop.details->arg1);
      addInstr(env, RISCV64Instr_FpBinary(op, dst, src1, src2));
      return dst;
   }

   /* ---------------------- BINARY OP ---------------------- */
   case Iex_Binop: {
      switch (e->Iex.Binop.op) {
      case Iop_SqrtF32: {
         HReg dst = newVRegF(env);
         HReg src = iselFltExpr(env, e->Iex.Binop.arg2);
         set_fcsr_rounding_mode(env, e->Iex.Binop.arg1);
         addInstr(env, RISCV64Instr_FpUnary(RISCV64op_FSQRT_S, dst, src));
         return dst;
      }
      case Iop_SqrtF64: {
         HReg dst = newVRegF(env);
         HReg src = iselFltExpr(env, e->Iex.Binop.arg2);
         set_fcsr_rounding_mode(env, e->Iex.Binop.arg1);
         addInstr(env, RISCV64Instr_FpUnary(RISCV64op_FSQRT_D, dst, src));
         return dst;
      }
      case Iop_I32StoF32:
      case Iop_I32UtoF32:
      case Iop_I64StoF32:
      case Iop_I64UtoF32:
      case Iop_I64StoF64:
      case Iop_I64UtoF64: {
         RISCV64FpConvertOp op;
         switch (e->Iex.Binop.op) {
         case Iop_I32StoF32:
            op = RISCV64op_FCVT_S_W;
            break;
         case Iop_I32UtoF32:
            op = RISCV64op_FCVT_S_WU;
            break;
         case Iop_I64StoF32:
            op = RISCV64op_FCVT_S_L;
            break;
         case Iop_I64UtoF32:
            op = RISCV64op_FCVT_S_LU;
            break;
         case Iop_I64StoF64:
            op = RISCV64op_FCVT_D_L;
            break;
         case Iop_I64UtoF64:
            op = RISCV64op_FCVT_D_LU;
            break;
         default:
            vassert(0);
         }
         HReg dst = newVRegF(env);
         HReg src = iselIntExpr_R(env, e->Iex.Binop.arg2);
         set_fcsr_rounding_mode(env, e->Iex.Binop.arg1);
         addInstr(env, RISCV64Instr_FpConvert(op, dst, src));
         return dst;
      }
      case Iop_F64toF32: {
         HReg dst = newVRegF(env);
         HReg src = iselFltExpr(env, e->Iex.Binop.arg2);
         set_fcsr_rounding_mode(env, e->Iex.Binop.arg1);
         addInstr(env, RISCV64Instr_FpConvert(RISCV64op_FCVT_S_D, dst, src));
         return dst;
      }
      case Iop_MinNumF32:
      case Iop_MaxNumF32:
      case Iop_MinNumF64:
      case Iop_MaxNumF64: {
         RISCV64FpBinaryOp op;
         switch (e->Iex.Binop.op) {
         case Iop_MinNumF32:
            op = RISCV64op_FMIN_S;
            break;
         case Iop_MaxNumF32:
            op = RISCV64op_FMAX_S;
            break;
         case Iop_MinNumF64:
            op = RISCV64op_FMIN_D;
            break;
         case Iop_MaxNumF64:
            op = RISCV64op_FMAX_D;
            break;
         default:
            vassert(0);
         }
         HReg dst  = newVRegF(env);
         HReg src1 = iselFltExpr(env, e->Iex.Binop.arg1);
         HReg src2 = iselFltExpr(env, e->Iex.Binop.arg2);
         addInstr(env, RISCV64Instr_FpBinary(op, dst, src1, src2));
         return dst;
      }
      default:
         break;
      }

      break;
   }

   /* ---------------------- UNARY OP ----------------------- */
   case Iex_Unop: {
      switch (e->Iex.Unop.op) {
      case Iop_NegF32:
      case Iop_AbsF32:
      case Iop_NegF64:
      case Iop_AbsF64: {
         RISCV64FpBinaryOp op;
         switch (e->Iex.Unop.op) {
         case Iop_NegF32:
            op = RISCV64op_FSGNJN_S;
            break;
         case Iop_AbsF32:
            op = RISCV64op_FSGNJX_S;
            break;
         case Iop_NegF64:
            op = RISCV64op_FSGNJN_D;
            break;
         case Iop_AbsF64:
            op = RISCV64op_FSGNJX_D;
            break;
         default:
            vassert(0);
         }
         HReg dst = newVRegF(env);
         HReg src = iselFltExpr(env, e->Iex.Unop.arg);
         addInstr(env, RISCV64Instr_FpBinary(op, dst, src, src));
         return dst;
      }
      case Iop_I32StoF64: {
         HReg dst = newVRegF(env);
         HReg src = iselIntExpr_R(env, e->Iex.Unop.arg);
         addInstr(env, RISCV64Instr_FpConvert(RISCV64op_FCVT_D_W, dst, src));
         return dst;
      }
      case Iop_I32UtoF64: {
         HReg dst = newVRegF(env);
         HReg src = iselIntExpr_R(env, e->Iex.Unop.arg);
         addInstr(env, RISCV64Instr_FpConvert(RISCV64op_FCVT_D_WU, dst, src));
         return dst;
      }
      case Iop_F32toF64: {
         HReg dst = newVRegF(env);
         HReg src = iselFltExpr(env, e->Iex.Unop.arg);
         addInstr(env, RISCV64Instr_FpConvert(RISCV64op_FCVT_D_S, dst, src));
         return dst;
      }
      case Iop_ReinterpI32asF32: {
         HReg dst = newVRegF(env);
         HReg src = iselIntExpr_R(env, e->Iex.Unop.arg);
         addInstr(env, RISCV64Instr_FpMove(RISCV64op_FMV_W_X, dst, src));
         return dst;
      }
      case Iop_ReinterpI64asF64: {
         HReg dst = newVRegF(env);
         HReg src = iselIntExpr_R(env, e->Iex.Unop.arg);
         addInstr(env, RISCV64Instr_FpMove(RISCV64op_FMV_D_X, dst, src));
         return dst;
      }
      default:
         break;
      }

      break;
   }

   /* ------------------------- GET ------------------------- */
   case Iex_Get: {
      HReg dst  = newVRegF(env);
      HReg base = get_baseblock_register();
      Int  off  = e->Iex.Get.offset - BASEBLOCK_OFFSET_ADJUSTMENT;
      vassert(off >= -2048 && off < 2048);

      if (ty == Ity_F32)
         addInstr(env, RISCV64Instr_FpLdSt(RISCV64op_FLW, dst, base, off));
      else if (ty == Ity_F64)
         addInstr(env, RISCV64Instr_FpLdSt(RISCV64op_FLD, dst, base, off));
      else
         vassert(0);
      return dst;
   }

   default:
      break;
   }

irreducible:
   ppIRExpr(e);
   vpanic("iselFltExpr(riscv64)");
}

/* Compute a floating-point value into a register, the identity of which is
   returned. As with iselIntExpr_R, the register will be virtual and must not be
   changed by subsequent code emitted by the caller. */
static HReg iselFltExpr(ISelEnv* env, IRExpr* e)
{
   HReg r = iselFltExpr_wrk(env, e);

   /* Sanity checks ... */
   vassert(hregClass(r) == HRcFlt64);
   vassert(hregIsVirtual(r));

   return r;
}

/*------------------------------------------------------------*/
/*--- ISEL: Statements                                     ---*/
/*------------------------------------------------------------*/

static void iselStmt(ISelEnv* env, IRStmt* stmt)
{
   if (vex_traceflags & VEX_TRACE_VCODE) {
      vex_printf("\n-- ");
      ppIRStmt(stmt);
      vex_printf("\n");
   }

   switch (stmt->tag) {
   /* ----------------------- LoadG ------------------------ */
   case Ist_LoadG: {
      IRLoadG* lg = stmt->Ist.LoadG.details;
      if (lg->end != Iend_LE)
         goto stmt_fail;

      IRType tyd = typeOfIRExpr(env->type_env, lg->alt);
      if (tyd == Ity_I64 || tyd == Ity_I32 || tyd == Ity_I16 || tyd == Ity_I8) {
         HReg dst = lookupIRTemp(env, lg->dst);
         HReg addr = iselIntExpr_R(env, lg->addr);
         HReg guard = iselIntExpr_R(env, lg->guard);
         HReg alt = iselIntExpr_R(env, lg->alt);

         vassert(lg->cvt == ILGop_Ident8 || lg->cvt == ILGop_Ident16 ||
                 lg->cvt == ILGop_Ident32 || lg->cvt == ILGop_Ident64);

         if (tyd == Ity_I64)
            addInstr(env, RISCV64Instr_LoadG(RISCV64op_LD, dst, addr, 0, guard, alt));
         else if (tyd == Ity_I32)
            addInstr(env, RISCV64Instr_LoadG(RISCV64op_LW, dst, addr, 0, guard, alt));
         else if (tyd == Ity_I16)
            addInstr(env, RISCV64Instr_LoadG(RISCV64op_LH, dst, addr, 0, guard, alt));
         else if (tyd == Ity_I8)
            addInstr(env, RISCV64Instr_LoadG(RISCV64op_LB, dst, addr, 0, guard, alt));
         else
            vassert(0);
         return;
      }
      return;
   }

   /* ------------------------ STORE ------------------------ */
   /* Little-endian write to memory. */
   case Ist_Store: {
      IRType tyd = typeOfIRExpr(env->type_env, stmt->Ist.Store.data);
      if (tyd == Ity_I64 || tyd == Ity_I32 || tyd == Ity_I16 || tyd == Ity_I8) {
         HReg src = iselIntExpr_R(env, stmt->Ist.Store.data);
         /* TODO Optimize the cases with small imm Add64/Sub64. */
         HReg addr = iselIntExpr_R(env, stmt->Ist.Store.addr);

         if (tyd == Ity_I64)
            addInstr(env, RISCV64Instr_Store(RISCV64op_SD, src, addr, 0));
         else if (tyd == Ity_I32)
            addInstr(env, RISCV64Instr_Store(RISCV64op_SW, src, addr, 0));
         else if (tyd == Ity_I16)
            addInstr(env, RISCV64Instr_Store(RISCV64op_SH, src, addr, 0));
         else if (tyd == Ity_I8)
            addInstr(env, RISCV64Instr_Store(RISCV64op_SB, src, addr, 0));
         else
            vassert(0);
         return;
      }
      if (tyd == Ity_F32 || tyd == Ity_F64) {
         HReg src  = iselFltExpr(env, stmt->Ist.Store.data);
         HReg addr = iselIntExpr_R(env, stmt->Ist.Store.addr);

         if (tyd == Ity_F32)
            addInstr(env, RISCV64Instr_FpLdSt(RISCV64op_FSW, src, addr, 0));
         else if (tyd == Ity_F64)
            addInstr(env, RISCV64Instr_FpLdSt(RISCV64op_FSD, src, addr, 0));
         else
            vassert(0);
         return;
      }
      break;
   }

   /* ----------------------- StoreG ------------------------ */
   case Ist_StoreG: {
      IRStoreG* sg = stmt->Ist.StoreG.details;
      if (sg->end != Iend_LE)
         goto stmt_fail;

      IRType tyd = typeOfIRExpr(env->type_env, sg->data);
      if (tyd == Ity_I64 || tyd == Ity_I32 || tyd == Ity_I16 || tyd == Ity_I8) {
         HReg src = iselIntExpr_R(env, sg->data);
         HReg addr = iselIntExpr_R(env, sg->addr);
         HReg guard = iselIntExpr_R(env, sg->guard);

         if (tyd == Ity_I64)
            addInstr(env, RISCV64Instr_StoreG(RISCV64op_SD, src, addr, 0, guard));
         else if (tyd == Ity_I32)
            addInstr(env, RISCV64Instr_StoreG(RISCV64op_SW, src, addr, 0, guard));
         else if (tyd == Ity_I16)
            addInstr(env, RISCV64Instr_StoreG(RISCV64op_SH, src, addr, 0, guard));
         else if (tyd == Ity_I8)
            addInstr(env, RISCV64Instr_StoreG(RISCV64op_SB, src, addr, 0, guard));
         else
            vassert(0);
         return;
      }
      return;
   }

   /* ------------------------- PUT ------------------------- */
   /* Write guest state, fixed offset. */
   case Ist_Put: {
      IRType tyd = typeOfIRExpr(env->type_env, stmt->Ist.Put.data);
      if (tyd == Ity_I64 || tyd == Ity_I32 || tyd == Ity_I16 || tyd == Ity_I8) {
         HReg src  = iselIntExpr_R(env, stmt->Ist.Put.data);
         HReg base = get_baseblock_register();
         Int  off  = stmt->Ist.Put.offset - BASEBLOCK_OFFSET_ADJUSTMENT;
         vassert(off >= -2048 && off < 2048);

         if (tyd == Ity_I64)
            addInstr(env, RISCV64Instr_Store(RISCV64op_SD, src, base, off));
         else if (tyd == Ity_I32)
            addInstr(env, RISCV64Instr_Store(RISCV64op_SW, src, base, off));
         else if (tyd == Ity_I16)
            addInstr(env, RISCV64Instr_Store(RISCV64op_SH, src, base, off));
         else if (tyd == Ity_I8)
            addInstr(env, RISCV64Instr_Store(RISCV64op_SB, src, base, off));
         else
            vassert(0);
         return;
      }
      if (tyd == Ity_F32 || tyd == Ity_F64) {
         HReg src  = iselFltExpr(env, stmt->Ist.Put.data);
         HReg base = get_baseblock_register();
         Int  off  = stmt->Ist.Put.offset - BASEBLOCK_OFFSET_ADJUSTMENT;
         vassert(off >= -2048 && off < 2048);

         if (tyd == Ity_F32)
            addInstr(env, RISCV64Instr_FpLdSt(RISCV64op_FSW, src, base, off));
         else if (tyd == Ity_F64)
            addInstr(env, RISCV64Instr_FpLdSt(RISCV64op_FSD, src, base, off));
         else
            vassert(0);
         return;
      }
      IRType vty = tyd & IR_TYPE_MASK;
      if (vty >= Ity_VLen1 && vty <= Ity_VLen64) {
         //Int vl = VLofVecIRType(tyd);
         Int sz = sizeofVecIRType(tyd);
         Int vlen_b = VLEN / 8;
         Int nregs = DIV_ROUND_UP(sz, vlen_b);
         Int vl_ldst64  = vlen_b / 8;

         HReg src[MAX_REGS] = {0};
         iselVecExpr_R(src, env, stmt->Ist.Put.data);

         Int  off  = stmt->Ist.Put.offset - BASEBLOCK_OFFSET_ADJUSTMENT;
         HReg base = newVRegI(env);

         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, base, get_baseblock_register(), off));
         addInstr(env, RISCV64Instr_VStore(RISCV64op_VStore64, vl_ldst64, src[0], base));

         for (int i = 1; i < nregs; ++i) {
            addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDI, base, base, vlen_b));
            addInstr(env, RISCV64Instr_VStore(RISCV64op_VStore64, vl_ldst64, src[i], base));
         }
         return;
      }
      break;
   }

   /* ------------------------- TMP ------------------------- */
   /* Assign value to temporary. */
   case Ist_WrTmp: {
      IRType ty = typeOfIRTemp(env->type_env, stmt->Ist.WrTmp.tmp);
      if (ty == Ity_I64 || ty == Ity_I32 || ty == Ity_I16 || ty == Ity_I8 ||
          ty == Ity_I1) {
         HReg dst = lookupIRTemp(env, stmt->Ist.WrTmp.tmp);
         HReg src = iselIntExpr_R(env, stmt->Ist.WrTmp.data);
         addInstr(env, RISCV64Instr_MV(dst, src));
         return;
      }
      if (ty == Ity_F32 || ty == Ity_F64) {
         HReg dst = lookupIRTemp(env, stmt->Ist.WrTmp.tmp);
         HReg src = iselFltExpr(env, stmt->Ist.WrTmp.data);
         addInstr(env, RISCV64Instr_FpMove(RISCV64op_FMV_D, dst, src));
         return;
      }
      IRType vty = ty & IR_TYPE_MASK;
      if (vty >= Ity_VLen1 && vty <= Ity_VLen64) {
         Int sz = sizeofVecIRType(ty);
         Int vlen_b = VLEN / 8;
         Int nregs = DIV_ROUND_UP(sz, vlen_b);

         HReg dst[MAX_REGS];
         HReg src[MAX_REGS];
         iselVecExpr_R(src, env, stmt->Ist.WrTmp.data);
         lookupIRTempVec(dst, nregs, env, stmt->Ist.WrTmp.tmp);
         for (int i = 0; i < nregs; ++i) {
            addInstr(env, RISCV64Instr_VMV(dst[i], src[i]));
         }
         return;
      }
      break;
   }

   /* ---------------- Call to DIRTY helper ----------------- */
   /* Call complex ("dirty") helper function. */
   case Ist_Dirty: {
      IRDirty* d = stmt->Ist.Dirty.details;

      /* Figure out the return type, if any. */
      IRType retty = Ity_INVALID;
      if (d->tmp != IRTemp_INVALID)
         retty = typeOfIRTemp(env->type_env, d->tmp);

      if (retty != Ity_INVALID && retty != Ity_I8 && retty != Ity_I16 &&
          retty != Ity_I32 && retty != Ity_I64)
         goto stmt_fail;

      /* Marshal args and do the call. */
      UInt   addToSp = 0;
      RetLoc rloc    = mk_RetLoc_INVALID();
      Bool   ok =
         doHelperCall(&addToSp, &rloc, env, d->guard, d->cee, retty, d->args);
      if (!ok)
         goto stmt_fail;
      vassert(is_sane_RetLoc(rloc));
      vassert(addToSp == 0);

      /* Now figure out what to do with the returned value, if any. */
      switch (retty) {
      case Ity_INVALID: {
         /* No return value. Nothing to do. */
         vassert(d->tmp == IRTemp_INVALID);
         vassert(rloc.pri == RLPri_None);
         return;
      }
      /* The returned value is for Ity_I<x> in x10/a0. Park it in the register
         associated with tmp. */
      case Ity_I8:
      case Ity_I16: {
         vassert(rloc.pri == RLPri_Int);
         /* Sign-extend the value returned from the helper as is expected by the
            rest of the backend. */
         HReg dst   = lookupIRTemp(env, d->tmp);
         UInt shift = 64 - 8 * sizeofIRType(retty);
         HReg tmp   = newVRegI(env);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SLLI, tmp,
                                           hregRISCV64_x10(), shift));
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_SRAI, dst, tmp, shift));
         return;
      }
      case Ity_I32: {
         vassert(rloc.pri == RLPri_Int);
         HReg dst = lookupIRTemp(env, d->tmp);
         addInstr(env, RISCV64Instr_ALUImm(RISCV64op_ADDIW, dst,
                                           hregRISCV64_x10(), 0));
         return;
      }
      case Ity_I64: {
         vassert(rloc.pri == RLPri_Int);
         HReg dst = lookupIRTemp(env, d->tmp);
         addInstr(env, RISCV64Instr_MV(dst, hregRISCV64_x10()));
         return;
      }
      default:
         vassert(0);
      }
      break;
   }

   /* ---------- Load Linked and Store Conditional ---------- */
   case Ist_LLSC: {
      if (stmt->Ist.LLSC.storedata == NULL) {
         /* LL */
         IRTemp res = stmt->Ist.LLSC.result;
         IRType ty  = typeOfIRTemp(env->type_env, res);
         if (ty == Ity_I32) {
            HReg r_dst  = lookupIRTemp(env, res);
            HReg r_addr = iselIntExpr_R(env, stmt->Ist.LLSC.addr);
            addInstr(env, RISCV64Instr_LoadR(RISCV64op_LR_W, r_dst, r_addr));
            return;
         }
      } else {
         /* SC */
         IRType tyd = typeOfIRExpr(env->type_env, stmt->Ist.LLSC.storedata);
         if (tyd == Ity_I32) {
            HReg r_tmp  = newVRegI(env);
            HReg r_src  = iselIntExpr_R(env, stmt->Ist.LLSC.storedata);
            HReg r_addr = iselIntExpr_R(env, stmt->Ist.LLSC.addr);
            addInstr(env,
                     RISCV64Instr_StoreC(RISCV64op_SC_W, r_tmp, r_src, r_addr));

            /* Now r_tmp is non-zero if failed, 0 if success. Change to IR
               conventions (0 is fail, 1 is success). */
            IRTemp res   = stmt->Ist.LLSC.result;
            HReg   r_res = lookupIRTemp(env, res);
            IRType ty    = typeOfIRTemp(env->type_env, res);
            vassert(ty == Ity_I1);
            addInstr(env,
                     RISCV64Instr_ALUImm(RISCV64op_SLTIU, r_res, r_tmp, 1));
            return;
         }
      }
      break;
   }

   /* ------------------------ ACAS ------------------------- */
   case Ist_CAS: {
      if (stmt->Ist.CAS.details->oldHi == IRTemp_INVALID) {
         /* "Normal" singleton CAS. */
         IRCAS* cas = stmt->Ist.CAS.details;
         IRType tyd = typeOfIRTemp(env->type_env, cas->oldLo);
         if (tyd == Ity_I64 || tyd == Ity_I32) {
            HReg old  = lookupIRTemp(env, cas->oldLo);
            HReg addr = iselIntExpr_R(env, cas->addr);
            HReg expd = iselIntExpr_R(env, cas->expdLo);
            HReg data = iselIntExpr_R(env, cas->dataLo);
            if (tyd == Ity_I64)
               addInstr(env, RISCV64Instr_CAS(RISCV64op_CAS_D, old, addr, expd,
                                              data));
            else
               addInstr(env, RISCV64Instr_CAS(RISCV64op_CAS_W, old, addr, expd,
                                              data));
            return;
         }
      }
      break;
   }

   /* ---------------------- MEM FENCE ---------------------- */
   case Ist_MBE:
      switch (stmt->Ist.MBE.event) {
      case Imbe_Fence:
         addInstr(env, RISCV64Instr_FENCE());
         return;
      default:
         break;
      }
      break;

   /* --------------------- INSTR MARK ---------------------- */
   /* Doesn't generate any executable code ... */
   case Ist_IMark:
      return;

   /* ---------------------- ABI HINT ----------------------- */
   /* These have no meaning (denotation in the IR) and so we ignore them ... if
      any actually made it this far. */
   case Ist_AbiHint:
       return;

   /* ------------------------ NO-OP ------------------------ */
   case Ist_NoOp:
      return;

   /* ------------------------ EXIT ------------------------- */
   case Ist_Exit: {
      if (stmt->Ist.Exit.dst->tag != Ico_U64)
         vpanic("iselStmt(riscv64): Ist_Exit: dst is not a 64-bit value");

      HReg cond   = iselIntExpr_R(env, stmt->Ist.Exit.guard);
      HReg base   = get_baseblock_register();
      Int  soff12 = stmt->Ist.Exit.offsIP - BASEBLOCK_OFFSET_ADJUSTMENT;
      vassert(soff12 >= -2048 && soff12 < 2048);

      /* Case: boring transfer to known address. */
      if (stmt->Ist.Exit.jk == Ijk_Boring) {
         if (env->chainingAllowed) {
            /* .. almost always true .. */
            /* Skip the event check at the dst if this is a forwards edge. */
            Bool toFastEP = (Addr64)stmt->Ist.Exit.dst->Ico.U64 > env->max_ga;
            if (0)
               vex_printf("%s", toFastEP ? "Y" : ",");
            addInstr(env, RISCV64Instr_XDirect(stmt->Ist.Exit.dst->Ico.U64,
                                               base, soff12, cond, toFastEP));
         } else {
            /* .. very occasionally .. */
            /* We can't use chaining, so ask for an assisted transfer, as
               that's the only alternative that is allowable. */
            HReg r = iselIntExpr_R(env, IRExpr_Const(stmt->Ist.Exit.dst));
            addInstr(env,
                     RISCV64Instr_XAssisted(r, base, soff12, cond, Ijk_Boring));
         }
         return;
      }

      /* Case: assisted transfer to arbitrary address. */
      switch (stmt->Ist.Exit.jk) {
      /* Keep this list in sync with that for iselNext below. */
      case Ijk_ClientReq:
      case Ijk_NoDecode:
      case Ijk_NoRedir:
      case Ijk_Sys_syscall:
      case Ijk_InvalICache:
      case Ijk_SigTRAP: {
         HReg r = iselIntExpr_R(env, IRExpr_Const(stmt->Ist.Exit.dst));
         addInstr(env, RISCV64Instr_XAssisted(r, base, soff12, cond,
                                              stmt->Ist.Exit.jk));
         return;
      }
      default:
         break;
      }

      /* Do we ever expect to see any other kind? */
      goto stmt_fail;
   }

   default:
      break;
   }

stmt_fail:
   ppIRStmt(stmt);
   vpanic("iselStmt");
}

/*------------------------------------------------------------*/
/*--- ISEL: Basic block terminators (Nexts)                ---*/
/*------------------------------------------------------------*/

static void iselNext(ISelEnv* env, IRExpr* next, IRJumpKind jk, Int offsIP)
{
   if (vex_traceflags & VEX_TRACE_VCODE) {
      vex_printf("\n-- PUT(%d) = ", offsIP);
      ppIRExpr(next);
      vex_printf("; exit-");
      ppIRJumpKind(jk);
      vex_printf("\n");
   }

   HReg base   = get_baseblock_register();
   Int  soff12 = offsIP - BASEBLOCK_OFFSET_ADJUSTMENT;
   vassert(soff12 >= -2048 && soff12 < 2048);

   /* Case: boring transfer to known address. */
   if (next->tag == Iex_Const) {
      IRConst* cdst = next->Iex.Const.con;
      vassert(cdst->tag == Ico_U64);
      if (jk == Ijk_Boring || jk == Ijk_Call) {
         /* Boring transfer to known address. */
         if (env->chainingAllowed) {
            /* .. almost always true .. */
            /* Skip the event check at the dst if this is a forwards edge. */
            Bool toFastEP = (Addr64)cdst->Ico.U64 > env->max_ga;
            if (0)
               vex_printf("%s", toFastEP ? "X" : ".");
            addInstr(env, RISCV64Instr_XDirect(cdst->Ico.U64, base, soff12,
                                               INVALID_HREG, toFastEP));
         } else {
            /* .. very occasionally .. */
            /* We can't use chaining, so ask for an assisted transfer, as that's
               the only alternative that is allowable. */
            HReg r = iselIntExpr_R(env, next);
            addInstr(env, RISCV64Instr_XAssisted(r, base, soff12, INVALID_HREG,
                                                 Ijk_Boring));
         }
         return;
      }
   }

   /* Case: call/return (==boring) transfer to any address. */
   switch (jk) {
   case Ijk_Boring:
   case Ijk_SyncupEnv:
   case Ijk_TooManyIR:
   case Ijk_Ret:
   case Ijk_Call: {
      HReg r = iselIntExpr_R(env, next);
      if (env->chainingAllowed)
         addInstr(env, RISCV64Instr_XIndir(r, base, soff12, INVALID_HREG));
      else
         addInstr(env, RISCV64Instr_XAssisted(r, base, soff12, INVALID_HREG,
                                              Ijk_Boring));
      return;
   }
   default:
      break;
   }

   /* Case: assisted transfer to arbitrary address. */
   switch (jk) {
   /* Keep this list in sync with that for Ist_Exit above. */
   case Ijk_ClientReq:
   case Ijk_NoDecode:
   case Ijk_NoRedir:
   case Ijk_Sys_syscall:
   case Ijk_InvalICache:
   case Ijk_SigTRAP: {
      HReg r = iselIntExpr_R(env, next);
      addInstr(env, RISCV64Instr_XAssisted(r, base, soff12, INVALID_HREG, jk));
      return;
   }
   default:
      break;
   }

   vex_printf("\n-- PUT(%d) = ", offsIP);
   ppIRExpr(next);
   vex_printf("; exit-");
   ppIRJumpKind(jk);
   vex_printf("\n");
   vassert(0); /* Are we expecting any other kind? */
}

/*------------------------------------------------------------*/
/*--- Insn selector top-level                              ---*/
/*------------------------------------------------------------*/

/* Translate an entire SB to riscv64 code. */

HInstrArray* iselSB_RISCV64(const IRSB*        bb,
                            VexArch            arch_host,
                            const VexArchInfo* archinfo_host,
                            const VexAbiInfo*  vbi,
                            Int                offs_Host_EvC_Counter,
                            Int                offs_Host_EvC_FailAddr,
                            Bool               chainingAllowed,
                            Bool               addProfInc,
                            Addr               max_ga)
{
   Int      i, j;
   HReg     hreg, hregHI;
   ISelEnv* env;

   /* Do some sanity checks. */
   vassert(arch_host == VexArchRISCV64);

   /* Check that the host's endianness is as expected. */
   vassert(archinfo_host->endness == VexEndnessLE);

   /* Guard against unexpected space regressions. */
   vassert(sizeof(RISCV64Instr) <= 32);

   /* Make up an initial environment to use. */
   env           = LibVEX_Alloc_inline(sizeof(ISelEnv));
   env->vreg_ctr = 0;

   /* Set up output code array. */
   env->code = newHInstrArray();

   /* Copy BB's type env. */
   env->type_env = bb->tyenv;

   /* Make up an IRTemp -> virtual HReg mapping. This doesn't change as we go
      along. */
   env->n_vregmap = bb->tyenv->types_used;
   env->vregmap   = LibVEX_Alloc_inline(env->n_vregmap * sizeof(HReg));
   env->vregmapHI = LibVEX_Alloc_inline(env->n_vregmap * sizeof(HReg));
   for (i = 0; i < 8; ++i) {
      env->vregmaps[i] = LibVEX_Alloc_inline(env->n_vregmap * sizeof(HReg));
   }

   /* and finally ... */
   env->chainingAllowed = chainingAllowed;
   env->hwcaps          = archinfo_host->hwcaps;
   env->previous_rm     = NULL;
   env->max_ga          = max_ga;

   /* For each IR temporary, allocate a suitably-kinded virtual register. */
   j = 0;
   for (i = 0; i < env->n_vregmap; i++) {
      hregHI = hreg = INVALID_HREG;
      switch (bb->tyenv->types[i] & IR_TYPE_MASK) {
      case Ity_I1:
      case Ity_I8:
      case Ity_I16:
      case Ity_I32:
      case Ity_I64:
         hreg = mkHReg(True, HRcInt64, 0, j++);
         break;
      case Ity_I128:
         hreg   = mkHReg(True, HRcInt64, 0, j++);
         hregHI = mkHReg(True, HRcInt64, 0, j++);
         break;
      case Ity_F32:
      case Ity_F64:
         hreg = mkHReg(True, HRcFlt64, 0, j++);
         break;
      case Ity_VLen1:
      case Ity_VLen8 ... Ity_VLen64:
         hreg = mkHReg(True, HRcVecVLen, 0, j++);
         hregHI = mkHReg(True, HRcVecVLen, 0, j++);
         for (int g = 2; g < 8; ++g) {
            env->vregmaps[g][i] = mkHReg(True, HRcVecVLen, 0, j++);
         }
         break;
      default:
         ppIRType(bb->tyenv->types[i]);
         vpanic("iselBB(riscv64): IRTemp type");
      }
      env->vregmap[i]   = hreg;
      env->vregmapHI[i] = hregHI;
   }
   env->vreg_ctr = j;

   /* The very first instruction must be an event check. */
   HReg base             = get_baseblock_register();
   Int  soff12_amCounter = offs_Host_EvC_Counter - BASEBLOCK_OFFSET_ADJUSTMENT;
   vassert(soff12_amCounter >= -2048 && soff12_amCounter < 2048);
   Int soff12_amFailAddr = offs_Host_EvC_FailAddr - BASEBLOCK_OFFSET_ADJUSTMENT;
   vassert(soff12_amFailAddr >= -2048 && soff12_amFailAddr < 2048);
   addInstr(env, RISCV64Instr_EvCheck(base, soff12_amCounter, base,
                                      soff12_amFailAddr));

   /* Possibly a block counter increment (for profiling). At this point we don't
      know the address of the counter, so just pretend it is zero. It will have
      to be patched later, but before this translation is used, by a call to
      LibVEX_PatchProfInc(). */
   if (addProfInc)
      addInstr(env, RISCV64Instr_ProfInc());

   /* Ok, finally we can iterate over the statements. */
   for (i = 0; i < bb->stmts_used; i++)
      iselStmt(env, bb->stmts[i]);

   iselNext(env, bb->next, bb->jumpkind, bb->offsIP);

   /* Record the number of vregs we used. */
   env->code->n_vregs = env->vreg_ctr;
   return env->code;
}

/*--------------------------------------------------------------------*/
/*--- end                                      host_riscv64_isel.c ---*/
/*--------------------------------------------------------------------*/
