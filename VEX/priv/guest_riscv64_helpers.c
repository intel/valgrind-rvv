
/*--------------------------------------------------------------------*/
/*--- begin                                guest_riscv64_helpers.c ---*/
/*--------------------------------------------------------------------*/

/*
   This file is part of Valgrind, a dynamic binary instrumentation
   framework.

   Copyright (C) 2020-2021 Petr Pavlu
      setup@dagobah.cz

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

#include "libvex_guest_riscv64.h"

#include "guest_riscv64_defs.h"
#include "main_util.h"

/* This file contains helper functions for riscv64 guest code. Calls to these
   functions are generated by the back end. These calls are of course in the
   host machine code and this file will be compiled to host machine code, so
   that all makes sense.

   Only change the signatures of these helper functions very carefully. If you
   change the signature here, you'll have to change the parameters passed to it
   in the IR calls constructed by guest_riscv64_toIR.c.

   The convention used is that all functions called from generated code are
   named riscv64g_<something>, and any function whose name lacks that prefix is
   not called from generated code. Note that some LibVEX_* functions can however
   be called by VEX's client, but that is not the same as calling them from
   VEX-generated code.
*/

#if defined(__riscv) && (__riscv_xlen == 64)
/* clang-format off */
#define CALCULATE_FFLAGS_BINARY64(inst)                                        \
   do {                                                                        \
      UInt res;                                                                \
      __asm__ __volatile__("csrr t0, fcsr\n\t"                                 \
                           "csrw frm, %[rm]\n\t"                               \
                           "csrw fflags, zero\n\t"                             \
                           inst " %[a1], %[a1], %[a2]\n\t"                     \
                           "csrr %[res], fflags\n\t"                           \
                           "csrw fcsr, t0\n\t"                                 \
                           : [res] "=r"(res)                                   \
                           : [a1] "f"(a1), [a2] "f"(a2), [rm] "r"(rm_RISCV)    \
                           : "t0");                                            \
      return res;                                                              \
   } while (0)
/* clang-format on */
#else
/* No simulated version is currently implemented. */
#define CALCULATE_FFLAGS_BINARY64(inst)                                        \
   do {                                                                        \
      return 0;                                                                \
   } while (0)
#endif

/* CALLED FROM GENERATED CODE: CLEAN HELPERS */
UInt riscv64g_calculate_fflags_fadd_d(Double a1, Double a2, UInt rm_RISCV)
{
   CALCULATE_FFLAGS_BINARY64("fadd.d");
}
UInt riscv64g_calculate_fflags_fsub_d(Double a1, Double a2, UInt rm_RISCV)
{
   CALCULATE_FFLAGS_BINARY64("fsub.d");
}
UInt riscv64g_calculate_fflags_fmul_d(Double a1, Double a2, UInt rm_RISCV)
{
   CALCULATE_FFLAGS_BINARY64("fmul.d");
}
UInt riscv64g_calculate_fflags_fdiv_d(Double a1, Double a2, UInt rm_RISCV)
{
   CALCULATE_FFLAGS_BINARY64("fdiv.d");
}

/*------------------------------------------------------------*/
/*--- Flag-helpers translation-time function specialisers. ---*/
/*--- These help iropt specialise calls the above run-time ---*/
/*--- flags functions.                                     ---*/
/*------------------------------------------------------------*/

IRExpr* guest_riscv64_spechelper(const HChar* function_name,
                                 IRExpr**     args,
                                 IRStmt**     precedingStmts,
                                 Int          n_precedingStmts)
{
   return NULL;
}

/*------------------------------------------------------------*/
/*--- Helpers for dealing with, and describing, guest      ---*/
/*--- state as a whole.                                    ---*/
/*------------------------------------------------------------*/

/* Initialise the entire riscv64 guest state. */
/* VISIBLE TO LIBVEX CLIENT */
void LibVEX_GuestRISCV64_initialise(/*OUT*/ VexGuestRISCV64State* vex_state)
{
   vex_bzero(vex_state, sizeof(*vex_state));
}

/* Figure out if any part of the guest state contained in minoff .. maxoff
   requires precise memory exceptions. If in doubt return True (but this
   generates significantly slower code).

   By default we enforce precise exns for guest x2 (sp), x8 (fp) and pc only.
   These are the minimum needed to extract correct stack backtraces from riscv64
   code.

   Only x2 (sp) is needed in mode VexRegUpdSpAtMemAccess.
*/
Bool guest_riscv64_state_requires_precise_mem_exns(Int                minoff,
                                                   Int                maxoff,
                                                   VexRegisterUpdates pxControl)
{
   Int fp_min = offsetof(VexGuestRISCV64State, guest_x8);
   Int fp_max = fp_min + 8 - 1;
   Int sp_min = offsetof(VexGuestRISCV64State, guest_x2);
   Int sp_max = sp_min + 8 - 1;
   Int pc_min = offsetof(VexGuestRISCV64State, guest_pc);
   Int pc_max = pc_min + 8 - 1;

   if (maxoff < sp_min || minoff > sp_max) {
      /* No overlap with sp. */
      if (pxControl == VexRegUpdSpAtMemAccess)
         return False; /* We only need to check stack pointer. */
   } else
      return True;

   if (maxoff < fp_min || minoff > fp_max) {
      /* No overlap with fp. */
   } else
      return True;

   if (maxoff < pc_min || minoff > pc_max) {
      /* No overlap with pc. */
   } else
      return True;

   return False;
}

#define ALWAYSDEFD(field)                                                      \
   {                                                                           \
      offsetof(VexGuestRISCV64State, field),                                   \
         (sizeof((VexGuestRISCV64State*)0)->field)                             \
   }

VexGuestLayout riscv64guest_layout = {
   /* Total size of the guest state, in bytes. */
   .total_sizeB = sizeof(VexGuestRISCV64State),

   /* Describe the stack pointer. */
   .offset_SP = offsetof(VexGuestRISCV64State, guest_x2),
   .sizeof_SP = 8,

   /* Describe the frame pointer. */
   .offset_FP = offsetof(VexGuestRISCV64State, guest_x8),
   .sizeof_FP = 8,

   /* Describe the instruction pointer. */
   .offset_IP = offsetof(VexGuestRISCV64State, guest_pc),
   .sizeof_IP = 8,

   /* Describe any sections to be regarded by Memcheck as 'always-defined'. */
   .n_alwaysDefd = 6,

   .alwaysDefd = {
      /* 0 */ ALWAYSDEFD(guest_x0),
      /* 1 */ ALWAYSDEFD(guest_pc),
      /* 2 */ ALWAYSDEFD(guest_EMNOTE),
      /* 3 */ ALWAYSDEFD(guest_CMSTART),
      /* 4 */ ALWAYSDEFD(guest_CMLEN),
      /* 5 */ ALWAYSDEFD(guest_NRADDR),
   },
};

/*--------------------------------------------------------------------*/
/*--- end                                  guest_riscv64_helpers.c ---*/
/*--------------------------------------------------------------------*/
