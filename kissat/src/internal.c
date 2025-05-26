#include "allocate.h"
#include "backtrack.h"
#include "error.h"
#include "search.h"
#include "import.h"
#include "inline.h"
#include "inlineframes.h"
#include "print.h"
#include "propsearch.h"
#include "require.h"
#include "resize.h"
#include "resources.h"

#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>

kissat *
kissat_init (void)
{
  kissat *solver = kissat_calloc (0, 1, sizeof *solver);
#ifndef NOPTIONS
  kissat_init_options (&solver->options);
#else
  kissat_init_options ();
#endif
#ifndef QUIET
  kissat_init_profiles (&solver->profiles);
#endif
  START (total);
  kissat_init_queue (solver);
  assert (INTERNAL_MAX_LIT < UINT_MAX);
  kissat_push_frame (solver, UINT_MAX);
  solver->watching = true;
  solver->conflict.size = 2;
  solver->conflict.keep = true;
  solver->scinc = 1.0;
  solver->first_reducible = INVALID_REF;
  solver->last_irredundant = INVALID_REF;
#ifndef NDEBUG
  kissat_init_checker (solver);
#endif
  return solver;
}

#define DEALLOC_GENERIC(NAME, ELEMENTS_PER_BLOCK) \
do { \
  const size_t block_size = ELEMENTS_PER_BLOCK * sizeof *solver->NAME; \
  kissat_dealloc (solver, solver->NAME, solver->size, block_size); \
  solver->NAME = 0; \
} while (0)

#define DEALLOC_VARIABLE_INDEXED(NAME) \
  DEALLOC_GENERIC (NAME, 1)

#define DEALLOC_LITERAL_INDEXED(NAME) \
  DEALLOC_GENERIC (NAME, 2)

#define RELEASE_LITERAL_INDEXED_STACKS(NAME,ACCESS) \
do { \
  for (all_stack (unsigned, IDX_RILIS, solver->active)) \
    { \
      const unsigned LIT_RILIS = LIT (IDX_RILIS); \
      const unsigned NOT_LIT_RILIS = NOT (LIT_RILIS); \
      RELEASE_STACK (ACCESS (LIT_RILIS)); \
      RELEASE_STACK (ACCESS (NOT_LIT_RILIS)); \
    } \
  DEALLOC_LITERAL_INDEXED (NAME); \
} while (0)

void save_status_to_file(kissat * solver, const char *filename)
{
  // 打开文件，使用追加模式
  FILE *file = fopen(filename, "a");
  if (!file) {
    fprintf(stderr, "Error: Could not open file %s for writing\n", filename);
    return;
  }
  // fprintf(file, "%u ", solver->backbone_computing?1:0);
  fprintf(file, "%u ", solver->extended?1:0);
  fprintf(file, "%u ", solver->inconsistent?1:0);
  fprintf(file, "%u ", solver->iterating?1:0);
  fprintf(file, "%u ", solver->probing?1:0);
  fprintf(file, "%u ", solver->stable?1:0);
  fprintf(file, "%u ", solver->watching?1:0);
  fprintf(file, "%u ", solver->large_clauses_watched_after_binary_clauses?1:0);
  fprintf(file, "%u ", solver->vars);
  fprintf(file, "%u ", solver->size);
  fprintf(file, "%u ", solver->active);

  // imports import 
  const import *imp_p = END_STACK (solver->import);
  const import *const imp_begin = BEGIN_STACK (solver->import);
  if (imp_p != imp_begin) {
    const import imp = *--imp_p;
    fprintf(file, "%u ", imp.lit);
    fprintf(file, "%u ", imp.imported?1:0);
    fprintf(file, "%u ", imp.eliminated?1:0);
  }
  else {
    fprintf(file, "-1 -1 -1 ");
  }

  // extensions extend
  const extension *ext_p = END_STACK (solver->extend);
  const extension *const ext_begin = BEGIN_STACK (solver->extend);
  if (ext_p != ext_begin) {
    const extension ext = *--ext_p;
    fprintf(file, "%u ", ext.lit);
    fprintf(file, "%u ", ext.blocking?1:0);
  }
  else {
    fprintf(file, "-1 -1 ");
  }

  // assigned 
  fprintf(file, "%u ", solver->assigned->level);
  fprintf(file, "%u ", solver->assigned->trail);
  fprintf(file, "%u ", solver->assigned->analyzed?1:0);
  fprintf(file, "%u ", solver->assigned->binary);
  fprintf(file, "%u ", solver->assigned->poisoned?1:0);
  fprintf(file, "%u ", solver->assigned->redundant?1:0);
  fprintf(file, "%u ", solver->assigned->removable?1:0);
  fprintf(file, "%u ", solver->assigned->shrinkable?1:0);
  fprintf(file, "%u ", solver->assigned->reason);

  // flags
  fprintf(file, "%u ", solver->flags->active?1:0);
  fprintf(file, "%u ", solver->flags->backbone0?1:0);
  fprintf(file, "%u ", solver->flags->backbone1?1:0);
  fprintf(file, "%u ", solver->flags->eliminate?1:0);
  fprintf(file, "%u ", solver->flags->eliminated?1:0);
  fprintf(file, "%u ", solver->flags->fixed?1:0);
  fprintf(file, "%u ", solver->flags->subsume?1:0);
  fprintf(file, "%u ", solver->flags->sweep?1:0);

  // queue 
  fprintf(file, "%u ", solver->queue.first);
  fprintf(file, "%u ", solver->queue.last);
  fprintf(file, "%u ", solver->queue.stamp);
  fprintf(file, "%u ", solver->queue.search.idx);
  fprintf(file, "%u ", solver->queue.search.stamp);

  // heap 
  fprintf(file, "%u ", solver->scores.tainted?1:0);
  fprintf(file, "%u ", solver->scores.vars);
  fprintf(file, "%u ", solver->scores.size);
  fprintf(file, "%lf ", *(solver->scores.score));
  fprintf(file, "%u ", *(solver->scores.pos));
  
  fprintf(file, "%lf ", solver->scinc);

  fprintf(file, "%u ", solver->level);

  fprintf(file, "%u ", *(solver->propagate));

  fprintf(file, "%u ", solver->best_assigned);
  fprintf(file, "%u ", solver->target_assigned);
  fprintf(file, "%u ", solver->unflushed);
  fprintf(file, "%u ", solver->unassigned);

  fprintf(file, "%u ", solver->resolvent_size);
  fprintf(file, "%u ", solver->antecedent_size);

  fprintf(file, "%u ", solver->clause_satisfied?1:0);
  fprintf(file, "%u ", solver->clause_shrink?1:0);
  fprintf(file, "%u ", solver->clause_trivial?1:0);

  fprintf(file, "%u ", solver->first_reducible);
  fprintf(file, "%u ", solver->last_irredundant);

  // averages[2]
  fprintf(file, "%u ", solver->averages[0].initialized?1:0);
  fprintf(file, "%lf ", solver->averages[0].fast_glue.value);
  fprintf(file, "%lf ", solver->averages[0].fast_glue.biased);
  fprintf(file, "%lf ", solver->averages[0].fast_glue.alpha);
  fprintf(file, "%lf ", solver->averages[0].fast_glue.beta);
  fprintf(file, "%lf ", solver->averages[0].fast_glue.exp);
  fprintf(file, "%lf ", solver->averages[0].slow_glue.value);
  fprintf(file, "%lf ", solver->averages[0].slow_glue.biased);
  fprintf(file, "%lf ", solver->averages[0].slow_glue.alpha);
  fprintf(file, "%lf ", solver->averages[0].slow_glue.beta);
  fprintf(file, "%lf ", solver->averages[0].slow_glue.exp);
  fprintf(file, "%lf ", solver->averages[0].decision_rate.value);
  fprintf(file, "%lf ", solver->averages[0].decision_rate.biased);
  fprintf(file, "%lf ", solver->averages[0].decision_rate.alpha);
  fprintf(file, "%lf ", solver->averages[0].decision_rate.beta);
  fprintf(file, "%lf ", solver->averages[0].decision_rate.exp);
  fprintf(file, "%llu ", solver->averages[0].saved_decisions);
  fprintf(file, "%u ", solver->averages[1].initialized?1:0);
  fprintf(file, "%lf ", solver->averages[1].fast_glue.value);
  fprintf(file, "%lf ", solver->averages[1].fast_glue.biased);
  fprintf(file, "%lf ", solver->averages[1].fast_glue.alpha);
  fprintf(file, "%lf ", solver->averages[1].fast_glue.beta);
  fprintf(file, "%lf ", solver->averages[1].fast_glue.exp);
  fprintf(file, "%lf ", solver->averages[1].slow_glue.value);
  fprintf(file, "%lf ", solver->averages[1].slow_glue.biased);
  fprintf(file, "%lf ", solver->averages[1].slow_glue.alpha);
  fprintf(file, "%lf ", solver->averages[1].slow_glue.beta);
  fprintf(file, "%lf ", solver->averages[1].slow_glue.exp);
  fprintf(file, "%lf ", solver->averages[1].decision_rate.value);
  fprintf(file, "%lf ", solver->averages[1].decision_rate.biased);
  fprintf(file, "%lf ", solver->averages[1].decision_rate.alpha);
  fprintf(file, "%lf ", solver->averages[1].decision_rate.beta);
  fprintf(file, "%lf ", solver->averages[1].decision_rate.exp);
  fprintf(file, "%llu ", solver->averages[1].saved_decisions);

  // reluctant
  fprintf(file, "%u ", solver->reluctant.limited?1:0);
  fprintf(file, "%u ", solver->reluctant.trigger?1:0);
  fprintf(file, "%llu ", (unsigned long long) solver->reluctant.period);
  fprintf(file, "%llu ", (unsigned long long) solver->reluctant.wait);
  fprintf(file, "%llu ", (unsigned long long) solver->reluctant.u);
  fprintf(file, "%llu ", (unsigned long long) solver->reluctant.v);
  fprintf(file, "%llu ", (unsigned long long) solver->reluctant.limit);

  // bounds
  fprintf(file, "%llu ", (unsigned long long) solver->bounds.eliminate.max_bound_completed);
  fprintf(file, "%u ", solver->bounds.eliminate.additional_clauses);

  // delays
  fprintf(file, "%u ", solver->delays.backbone.count);
  fprintf(file, "%u ", solver->delays.backbone.current);
  fprintf(file, "%u ", solver->delays.bumpreasons.count);
  fprintf(file, "%u ", solver->delays.bumpreasons.current);
  fprintf(file, "%u ", solver->delays.eliminate.count);
  fprintf(file, "%u ", solver->delays.eliminate.current);
  fprintf(file, "%u ", solver->delays.failed.count);
  fprintf(file, "%u ", solver->delays.failed.current);
  fprintf(file, "%u ", solver->delays.probe.count);
  fprintf(file, "%u ", solver->delays.probe.current);
  fprintf(file, "%u ", solver->delays.substitute.count);
  fprintf(file, "%u ", solver->delays.substitute.current);

  // enabled
  fprintf(file, "%u ", solver->enabled.eliminate?1:0);
  fprintf(file, "%u ", solver->enabled.focus?1:0);
  fprintf(file, "%u ", solver->enabled.mode?1:0);
  fprintf(file, "%u ", solver->enabled.probe?1:0);

  // effort 
  fprintf(file, "%llu ", (unsigned long long) solver->last.eliminate);
  fprintf(file, "%llu ", (unsigned long long) solver->last.probe);

  // limited 
  fprintf(file, "%u ", solver->limited.conflicts?1:0);
  fprintf(file, "%u ", solver->limited.decisions?1:0);

  // limits 
  fprintf(file, "%llu ", (unsigned long long) solver->limits.conflicts);
  fprintf(file, "%llu ", (unsigned long long) solver->limits.decisions);
  fprintf(file, "%llu ", (unsigned long long) solver->limits.reports);
  fprintf(file, "%llu ", (unsigned long long) solver->limits.mode.ticks);
  fprintf(file, "%llu ", (unsigned long long) solver->limits.mode.conflicts);
  fprintf(file, "%llu ", (unsigned long long) solver->limits.mode.interval);

  // waiting 
  fprintf(file, "%llu ", (unsigned long long) solver->waiting.eliminate.reduce);
  fprintf(file, "%llu ", (unsigned long long) solver->waiting.probe.reduce);

  // walked
  fprintf(file, "%u ", solver->walked);

  // statistics
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.allocated_collected);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.allocated_current);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.allocated_max);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.ands_eliminated);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.ands_extracted);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.arena_enlarged);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.arena_garbage);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.arena_resized);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.arena_shrunken);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.backbone_computations);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.backbone_implied);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.backbone_probes);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.backbone_propagations);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.backbone_rounds);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.backbone_ticks);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.backbone_units);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.best_saved);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.chronological);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.clauses_added);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.clauses_deleted);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.clauses_improved);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.clauses_irredundant);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.clauses_kept2);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.clauses_kept3);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.clauses_learned);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.clauses_original);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.clauses_promoted1);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.clauses_promoted2);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.clauses_reduced);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.clauses_redundant);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.compacted);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.conflicts);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.decisions);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.definitions_checked);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.definitions_eliminated);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.definitions_extracted);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.definition_units);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.defragmentations);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.dense_garbage_collections);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.dense_propagations);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.dense_ticks);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.duplicated);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.eliminated);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.eliminations);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.eliminate_attempted);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.eliminate_resolutions);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.eliminate_units);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.equivalences_eliminated);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.equivalences_extracted);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.extensions);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.flipped);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.flushed);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.focused_decisions);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.focused_modes);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.focused_propagations);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.focused_restarts);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.focused_ticks);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.forward_checks);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.forward_steps);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.forward_strengthened);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.forward_subsumed);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.forward_subsumptions);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.garbage_collections);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.gates_checked);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.gates_eliminated);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.gates_extracted);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.if_then_else_eliminated);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.if_then_else_extracted);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.initial_decisions);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.kitten_conflicts);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.kitten_decisions);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.kitten_flip);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.kitten_flipped);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.kitten_propagations);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.kitten_sat);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.kitten_solved);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.kitten_ticks);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.kitten_unknown);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.kitten_unsat);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.learned_units);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.literals_bumped);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.literals_deduced);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.literals_learned);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.literals_minimized);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.literals_minimize_shrunken);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.literals_shrunken);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.moved);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.on_the_fly_strengthened);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.on_the_fly_subsumed);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.probing_propagations);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.probings);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.probing_ticks);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.propagations);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.reductions);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.rephased);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.rephased_best);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.rephased_inverted);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.rephased_original);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.rephased_walking);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.rescaled);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.restarts);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.saved_decisions);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.searches);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.search_propagations);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.search_ticks);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.sparse_garbage_collections);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.stable_decisions);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.stable_modes);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.stable_propagations);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.stable_restarts);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.stable_ticks);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.strengthened);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.substituted);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.substitute_ticks);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.subsumption_checks);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.substitute_units);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.substitutions);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.subsumed);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.sweep);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.sweep_completed);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.sweep_equivalences);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.sweep_sat);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.sweep_solved);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.sweep_units);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.sweep_unsat);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.sweep_variables);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.switched_modes);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.target_decisions);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.target_saved);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.ticks);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.units);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.variables_activated);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.variables_added);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.variables_removed);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.vectors_defrags_needed);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.vectors_enlarged);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.vivifications);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.vivified);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.vivify_checks);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.vivify_implied);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.vivify_probes);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.vivify_propagations);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.vivify_reused);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.vivify_strengthened);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.vivify_subsumed);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.vivify_ticks);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.vivify_units);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.walk_decisions);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.walk_improved);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.walk_previous);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.walks);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.walk_steps);
  fprintf(file, "%llu ", (unsigned long long) solver->statistics.warmups);
  // fprintf(file, "%llu ", (unsigned long long) solver->statistics.weakened);

  // mode 
  fprintf(file, "%llu ", (unsigned long long) solver->mode.ticks);

  fprintf(file, "%llu ", (unsigned long long) solver->ticks);

  // format
  fprintf(file, "%u ", solver->format.pos);

  fprintf(file, "\n"); 
  fflush(file);
  fclose(file);
  return; 
}

void
kissat_release (kissat * solver)
{
  kissat_require_initialized (solver);
  kissat_release_heap (solver, SCORES);

  kissat_release_phases (solver);

  RELEASE_STACK (solver->export);
  RELEASE_STACK (solver->import);

  DEALLOC_VARIABLE_INDEXED (assigned);
  DEALLOC_VARIABLE_INDEXED (flags);
  DEALLOC_VARIABLE_INDEXED (links);

  DEALLOC_LITERAL_INDEXED (marks);
  DEALLOC_LITERAL_INDEXED (values);
  DEALLOC_LITERAL_INDEXED (watches);

  RELEASE_STACK (solver->import);
  RELEASE_STACK (solver->eliminated);
  RELEASE_STACK (solver->extend);
  RELEASE_STACK (solver->witness);
  RELEASE_STACK (solver->etrail);

  RELEASE_STACK (solver->vectors.stack);
  RELEASE_STACK (solver->delayed);

  RELEASE_STACK (solver->clause);
  RELEASE_STACK (solver->shadow);
#if defined(LOGGING) || !defined(NDEBUG)
  RELEASE_STACK (solver->resolvent);
#endif

  RELEASE_STACK (solver->arena);

  RELEASE_STACK (solver->units);
  RELEASE_STACK (solver->frames);
  RELEASE_STACK (solver->sorter);

  RELEASE_ARRAY (solver->trail, solver->size);

  RELEASE_STACK (solver->analyzed);
  RELEASE_STACK (solver->levels);
  RELEASE_STACK (solver->minimize);
  RELEASE_STACK (solver->poisoned);
  RELEASE_STACK (solver->promote);
  RELEASE_STACK (solver->removable);
  RELEASE_STACK (solver->shrinkable);
  RELEASE_STACK (solver->xorted[0]);
  RELEASE_STACK (solver->xorted[1]);

  RELEASE_STACK (solver->sweep);

  RELEASE_STACK (solver->ranks);

  RELEASE_STACK (solver->antecedents[0]);
  RELEASE_STACK (solver->antecedents[1]);
  RELEASE_STACK (solver->gates[0]);
  RELEASE_STACK (solver->gates[1]);
  RELEASE_STACK (solver->resolvents);

#if !defined(NDEBUG) || !defined(NPROOFS)
  RELEASE_STACK (solver->added);
  RELEASE_STACK (solver->removed);
#endif

#if !defined(NDEBUG) || !defined(NPROOFS) || defined(LOGGING)
  RELEASE_STACK (solver->original);
#endif

#ifndef QUIET
  RELEASE_STACK (solver->profiles.stack);
#endif

#ifndef NDEBUG
  kissat_release_checker (solver);
#endif
#if !defined(NDEBUG) && defined(METRICS)
  uint64_t leaked = solver->statistics.allocated_current;
  if (leaked)
    if (!getenv ("LEAK"))
      kissat_fatal ("internally leaking %" PRIu64 " bytes", leaked);
#endif

  kissat_free (0, solver, sizeof *solver);
}

void
kissat_reserve (kissat * solver, int max_var)
{
  kissat_require_initialized (solver);
  kissat_require (0 <= max_var,
		  "negative maximum variable argument '%u'", max_var);
  kissat_require (max_var <= EXTERNAL_MAX_VAR,
		  "invalid maximum variable argument '%u'", max_var);
  kissat_increase_size (solver, (unsigned) max_var);
}

int
kissat_get_option (kissat * solver, const char *name)
{
  kissat_require_initialized (solver);
  kissat_require (name, "name zero pointer");
#ifndef NOPTIONS
  return kissat_options_get (&solver->options, name);
#else
  (void) solver;
  return kissat_options_get (name);
#endif
}

int
kissat_set_option (kissat * solver, const char *name, int new_value)
{
#ifndef NOPTIONS
  kissat_require_initialized (solver);
  kissat_require (name, "name zero pointer");
#ifndef NOPTIONS
  return kissat_options_set (&solver->options, name, new_value);
#else
  return kissat_options_set (name, new_value);
#endif
#else
  (void) solver, (void) new_value;
  return kissat_options_get (name);
#endif
}

void
kissat_set_decision_limit (kissat * solver, unsigned limit)
{
  kissat_require_initialized (solver);
  limits *limits = &solver->limits;
  limited *limited = &solver->limited;
  statistics *statistics = &solver->statistics;
  limited->decisions = true;
  assert (UINT64_MAX - limit >= statistics->decisions);
  limits->decisions = statistics->decisions + limit;
  LOG ("set decision limit to %" PRIu64 " after %u decisions",
       limits->decisions, limit);
}

void
kissat_set_conflict_limit (kissat * solver, unsigned limit)
{
  kissat_require_initialized (solver);
  limits *limits = &solver->limits;
  limited *limited = &solver->limited;
  statistics *statistics = &solver->statistics;
  limited->conflicts = true;
  assert (UINT64_MAX - limit >= statistics->conflicts);
  limits->conflicts = statistics->conflicts + limit;
  LOG ("set conflict limit to %" PRIu64 " after %u conflicts",
       limits->conflicts, limit);
}

void
kissat_print_statistics (kissat * solver)
{
#ifndef QUIET
  kissat_require_initialized (solver);
  const int verbosity = kissat_verbosity (solver);
  if (verbosity < 0)
    return;
  if (GET_OPTION (profile))
    {
      kissat_section (solver, "profiling");
      kissat_profiles_print (solver);
    }
  const bool complete = GET_OPTION (statistics);
  kissat_section (solver, "statistics");
  const bool verbose = (complete || verbosity > 0);
  kissat_statistics_print (solver, verbose);
#ifndef NPROOFS
  if (solver->proof)
    {
      kissat_section (solver, "proof");
      kissat_print_proof_statistics (solver, verbose);
    }
#endif
#ifndef NDEBUG
  if (GET_OPTION (check) > 1)
    {
      kissat_section (solver, "checker");
      kissat_print_checker_statistics (solver, verbose);
    }
#endif
  kissat_section (solver, "resources");
  kissat_print_resources (solver);
#endif
  (void) solver;
}

void
kissat_add (kissat * solver, int elit)
{
  kissat_require_initialized (solver);
  kissat_require (!GET (searches), "incremental solving not supported");
#if !defined(NDEBUG) || !defined(NPROOFS) || defined(LOGGING)
  const int checking = kissat_checking (solver);
  const bool logging = kissat_logging (solver);
  const bool proving = kissat_proving (solver);
#endif
  if (elit)
    {
      kissat_require_valid_external_internal (elit);
#if !defined(NDEBUG) || !defined(NPROOFS) || defined(LOGGING)
      if (checking || logging || proving)
	PUSH_STACK (solver->original, elit);
#endif
      unsigned ilit = kissat_import_literal (solver, elit);

      const mark mark = MARK (ilit);
      if (!mark)
	{
	  const value value = kissat_fixed (solver, ilit);
	  if (value > 0)
	    {
	      if (!solver->clause_satisfied)
		{
		  LOG ("adding root level satisfied literal %u(%u)@0=1",
		       ilit, elit);
		  solver->clause_satisfied = true;
		}
	    }
	  else if (value < 0)
	    {
	      LOG ("adding root level falsified literal %u(%u)@0=-1",
		   ilit, elit);
	      if (!solver->clause_shrink)
		{
		  solver->clause_shrink = true;
		  LOG ("thus original clause needs shrinking");
		}
	    }
	  else
	    {
	      MARK (ilit) = 1;
	      MARK (NOT (ilit)) = -1;
	      assert (SIZE_STACK (solver->clause) < UINT_MAX);
	      PUSH_STACK (solver->clause, ilit);
	    }
	}
      else if (mark < 0)
	{
	  assert (mark < 0);
	  if (!solver->clause_trivial)
	    {
	      LOG ("adding dual literal %u(%u) and %u(%u)",
		   NOT (ilit), -elit, ilit, elit);
	      solver->clause_trivial = true;
	    }
	}
      else
	{
	  assert (mark > 0);
	  LOG ("adding duplicated literal %u(%u)", ilit, elit);
	  if (!solver->clause_shrink)
	    {
	      solver->clause_shrink = true;
	      LOG ("thus original clause needs shrinking");
	    }
	}
    }
  else
    {
#if !defined(NDEBUG) || !defined(NPROOFS) || defined(LOGGING)
      const size_t offset = solver->offset_of_last_original_clause;
      size_t esize = SIZE_STACK (solver->original) - offset;
      int *elits = BEGIN_STACK (solver->original) + offset;
      assert (esize <= UINT_MAX);
#endif
      ADD_UNCHECKED_EXTERNAL (esize, elits);
      const size_t isize = SIZE_STACK (solver->clause);
      unsigned *ilits = BEGIN_STACK (solver->clause);
      assert (isize < (unsigned) INT_MAX);

      if (solver->inconsistent)
	LOG ("inconsistent thus skipping original clause");
      else if (solver->clause_satisfied)
	LOG ("skipping satisfied original clause");
      else if (solver->clause_trivial)
	LOG ("skipping trivial original clause");
      else
	{
	  kissat_activate_literals (solver, isize, ilits);

	  if (!isize)
	    {
	      if (solver->clause_shrink)
		LOG ("all original clause literals root level falsified");
	      else
		LOG ("found empty original clause");

	      if (!solver->inconsistent)
		{
		  LOG ("thus solver becomes inconsistent");
		  solver->inconsistent = true;
		  CHECK_AND_ADD_EMPTY ();
		  ADD_EMPTY_TO_PROOF ();
		}
	    }
	  else if (isize == 1)
	    {
	      unsigned unit = TOP_STACK (solver->clause);

	      if (solver->clause_shrink)
		LOGUNARY (unit, "original clause shrinks to");
	      else
		LOGUNARY (unit, "found original");

	      kissat_original_unit (solver, unit);

	      COVER (solver->level);
	      if (!solver->level)
		(void) kissat_search_propagate (solver);
	    }
	  else
	    {
	      reference res = kissat_new_original_clause (solver);

	      const unsigned a = ilits[0];
	      const unsigned b = ilits[1];

	      const value u = VALUE (a);
	      const value v = VALUE (b);

	      const unsigned k = u ? LEVEL (a) : UINT_MAX;
	      const unsigned l = v ? LEVEL (b) : UINT_MAX;

	      bool assign = false;

	      if (!u && v < 0)
		{
		  LOG ("original clause immediately forcing");
		  assign = true;
		}
	      else if (u < 0 && k == l)
		{
		  LOG ("both watches falsified at level @%u", k);
		  assert (v < 0);
		  assert (k > 0);
		  kissat_backtrack_without_updating_phases (solver, k - 1);
		}
	      else if (u < 0)
		{
		  LOG ("watches falsified at levels @%u and @%u", k, l);
		  assert (v < 0);
		  assert (k > l);
		  assert (l > 0);
		  assign = true;
		}
	      else if (u > 0 && v < 0)
		{
		  LOG ("first watch satisfied at level @%u "
		       "second falsified at level @%u", k, l);
		  assert (k <= l);
		}
	      else if (!u && v > 0)
		{
		  LOG ("first watch unassigned "
		       "second falsified at level @%u", l);
		  assign = true;
		}
	      else
		{
		  assert (!u);
		  assert (!v);
		}

	      if (assign)
		{
		  assert (solver->level > 0);

		  if (isize == 2)
		    {
		      assert (res == INVALID_REF);
		      kissat_assign_binary (solver, false, a, b);
		    }
		  else
		    {
		      assert (res != INVALID_REF);
		      clause *c = kissat_dereference_clause (solver, res);
		      kissat_assign_reference (solver, a, res, c);
		    }
		}
	    }
	}

#if !defined(NDEBUG) || !defined(NPROOFS)
      if (solver->clause_satisfied || solver->clause_trivial)
	{
#ifndef NDEBUG
	  if (checking > 1)
	    kissat_remove_checker_external (solver, esize, elits);
#endif
#ifndef NPROOFS
	  if (proving)
	    {
	      if (esize == 1)
		LOG ("skipping deleting unit from proof");
	      else
		kissat_delete_external_from_proof (solver, esize, elits);
	    }
#endif
	}
      else if (!solver->inconsistent && solver->clause_shrink)
	{
#ifndef NDEBUG
	  if (checking > 1)
	    {
	      kissat_check_and_add_internal (solver, isize, ilits);
	      kissat_remove_checker_external (solver, esize, elits);
	    }
#endif
#ifndef NPROOFS
	  if (proving)
	    {
	      kissat_add_lits_to_proof (solver, isize, ilits);
	      kissat_delete_external_from_proof (solver, esize, elits);
	    }
#endif
	}
#endif

#if !defined(NDEBUG) || !defined(NPROOFS) || defined(LOGGING)
      if (checking)
	{
	  LOGINTS (esize, elits, "saved original");
	  PUSH_STACK (solver->original, 0);
	  solver->offset_of_last_original_clause =
	    SIZE_STACK (solver->original);
	}
      else if (logging || proving)
	{
	  LOGINTS (esize, elits, "reset original");
	  CLEAR_STACK (solver->original);
	  solver->offset_of_last_original_clause = 0;
	}
#endif
      for (all_stack (unsigned, lit, solver->clause))
	  MARK (lit) = MARK (NOT (lit)) = 0;

      CLEAR_STACK (solver->clause);

      solver->clause_satisfied = false;
      solver->clause_trivial = false;
      solver->clause_shrink = false;
    }
}

int
kissat_solve (kissat * solver)
{
  kissat_require_initialized (solver);
  kissat_require (EMPTY_STACK (solver->clause),
		  "incomplete clause (terminating zero not added)");
  kissat_require (!GET (searches), "incremental solving not supported");
  return kissat_search (solver);
}

void
kissat_terminate (kissat * solver)
{
  kissat_require_initialized (solver);
  solver->termination.flagged = ~(unsigned) 0;
  assert (solver->termination.flagged);
}

void
kissat_set_terminate (kissat * solver, void *state, int (*terminate) (void *))
{
  solver->termination.terminate = 0;
  solver->termination.state = state;
  solver->termination.terminate = terminate;
}

int
kissat_value (kissat * solver, int elit)
{
  kissat_require_initialized (solver);
  kissat_require_valid_external_internal (elit);
  const unsigned eidx = ABS (elit);
  if (eidx >= SIZE_STACK (solver->import))
    return 0;
  const import *const import = &PEEK_STACK (solver->import, eidx);
  if (!import->imported)
    return 0;
  value tmp;
  if (import->eliminated)
    {
      if (!solver->extended && !EMPTY_STACK (solver->extend))
	kissat_extend (solver);
      const unsigned eliminated = import->lit;
      tmp = PEEK_STACK (solver->eliminated, eliminated);
    }
  else
    {
      const unsigned ilit = import->lit;
      tmp = VALUE (ilit);
    }
  if (!tmp)
    return 0;
  if (elit < 0)
    tmp = -tmp;
  return tmp < 0 ? -elit : elit;
}
