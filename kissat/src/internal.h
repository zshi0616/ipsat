#ifndef _internal_h_INCLUDED
#define _internal_h_INCLUDED

#include "arena.h"
#include "array.h"
#include "assign.h"
#include "averages.h"
#include "check.h"
#include "clause.h"
#include "cover.h"
#include "extend.h"
#include "smooth.h"
#include "flags.h"
#include "format.h"
#include "frames.h"
#include "heap.h"
#include "kimits.h"
#include "kissat.h"
#include "literal.h"
#include "mode.h"
#include "options.h"
#include "phases.h"
#include "profile.h"
#include "proof.h"
#include "queue.h"
#include "random.h"
#include "reluctant.h"
#include "rephase.h"
#include "stack.h"
#include "statistics.h"
#include "literal.h"
#include "value.h"
#include "vector.h"
#include "watch.h"

typedef struct datarank datarank;

struct datarank
{
  unsigned data;
  unsigned rank;
};

typedef struct import import;

struct import
{
  unsigned lit:30;
  bool imported:1;
  bool eliminated:1;
};

typedef struct termination termination;

struct termination
{
#ifdef COVERAGE
  volatile uint64_t flagged;
#else
  volatile bool flagged;
#endif
  volatile void *state;
  int (*volatile terminate) (void *);
};

// *INDENT-OFF*

typedef STACK (value) eliminated;
typedef STACK (import) imports;
typedef STACK (datarank) dataranks;
typedef STACK (watch) statches;
typedef STACK (watch *) patches;

// *INDENT-ON*

struct kitten;

struct kissat
{
#if !defined(NDEBUG) || defined(METRICS)
  bool backbone_computing;
#endif
#ifdef LOGGING
  bool compacting;
#endif
  bool extended;
  bool inconsistent;
  bool iterating;
  bool probing;
#ifndef QUIET
  bool sectioned;
#endif
  bool stable;
#if !defined(NDEBUG) || defined(METRICS)
  bool vivifying;
#endif
  bool watching;

  bool large_clauses_watched_after_binary_clauses;

  termination termination;    // do not record

  unsigned vars;
  unsigned size;
  unsigned active;

  ints export;        // do not record
  ints units;         // do not record  
  imports import;
  extensions extend;
  unsigneds witness;  // do not record

  assigned *assigned;
  flags *flags;

  mark *marks;  // do not record

  value *values;  // do not record
  phases phases;  // do not record

  eliminated eliminated;  // do not record
  unsigneds etrail;   // do not record

  links *links;  // do not record
  queue queue;

  heap scores;
  double scinc;

  unsigned level;
  frames frames;  // do not record

  unsigned_array trail;  // do not record
  unsigned *propagate;

  unsigned best_assigned;
  unsigned target_assigned;
  unsigned unflushed;
  unsigned unassigned;

  unsigneds delayed;

#if defined(LOGGING) || !defined(NDEBUG)
  unsigneds resolvent;
#endif
  unsigned resolvent_size;
  unsigned antecedent_size;

  dataranks ranks;  // do not record

  unsigneds analyzed;  // do not record
  unsigneds levels;  // do not record
  unsigneds minimize;  // do not record
  unsigneds poisoned;  // do not record
  unsigneds promote;  // do not record
  unsigneds removable;  // do not record
  unsigneds shrinkable;  // do not record

  clause conflict;  // do not record

  bool clause_satisfied;
  bool clause_shrink;
  bool clause_trivial;

  unsigneds clause;  // do not record
  unsigneds shadow;  // do not record

  arena arena;  // do not record
  vectors vectors;  // do not record
  reference first_reducible;
  reference last_irredundant;
  watches *watches;  // do not record

  sizes sorter;  // do not record

  generator random;
  averages averages[2];
  reluctant reluctant;

  bounds bounds;
  delays delays;
  enabled enabled;
  effort last;
  limited limited;
  limits limits;
  waiting waiting;
  unsigned walked;

  statistics statistics;
  mode mode;

  uint64_t ticks;

  format format;

  statches antecedents[2];  // do not record
  statches gates[2];  // do not record
  patches xorted[2];  // do not record
  unsigneds resolvents;  // do not record
  bool resolve_gate;

  struct kitten *kitten;  // do not record
#ifdef METRICS
  uint64_t *gate_eliminated;
#else
  bool gate_eliminated;
#endif
  unsigneds sweep;  // do not record

#if !defined(NDEBUG) || !defined(NPROOFS)
  unsigneds added;
  unsigneds removed;
#endif

#if !defined(NDEBUG) || !defined(NPROOFS) || defined(LOGGING)
  ints original;
  size_t offset_of_last_original_clause;
#endif

#ifndef QUIET
  profiles profiles;
#endif

#ifndef NOPTIONS
  options options;
#endif

#ifndef NDEBUG
  checker *checker;
#endif

#ifndef NPROOFS
  proof *proof;
#endif
};

#define VARS (solver->vars)
#define LITS (2*solver->vars)

#define SCORES (&solver->scores)

static inline unsigned
kissat_assigned (kissat * solver)
{
  assert (VARS >= solver->unassigned);
  return VARS - solver->unassigned;
}

#define all_variables(IDX) \
  unsigned IDX = 0, IDX ## _END = solver->vars; \
  IDX != IDX ## _END; \
  ++IDX

#define all_literals(LIT) \
  unsigned LIT = 0, LIT ## _END = LITS; \
  LIT != LIT ## _END; \
  ++LIT

#define all_clauses(C) \
  clause *       C         = (clause*) BEGIN_STACK (solver->arena), \
         * const C ## _END = (clause*) END_STACK (solver->arena), \
	 * C ## _NEXT; \
  C != C ## _END && (C ## _NEXT = kissat_next_clause (C), true); \
  C = C ## _NEXT

#endif
