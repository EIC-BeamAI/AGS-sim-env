import os, json, argparse
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import pygad
import ags_gymnasium


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_generations", type=int, default=50000)
    p.add_argument("--num_parents_mating", type=int, default=2)
    p.add_argument("--sol_per_pop", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--h_tune", type=float, default=0.0)
    p.add_argument("--v_tune", type=float, default=0.0)
    p.add_argument("--misalign_sigma", type=float, default=1e-2)
    args = p.parse_args()

    base = "/work/data_science/monibor/DT4BNL/AGS-sim-env/ga/results_1"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = f"seed{args.seed}" if args.seed is not None else "noseed"

    outdir = os.path.join(
        base,
        f"ga_{ts}_gen{args.num_generations}_pop{args.sol_per_pop}"
        f"_par{args.num_parents_mating}_sigma{args.misalign_sigma}_{seed}"
    )
    os.makedirs(outdir)

    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    env = gym.make(
        "AGS-Injection-v0",
        h_tune=args.h_tune,
        v_tune=args.v_tune,
        misalign_sigma=args.misalign_sigma,
    )

    num_genes = env.action_space.shape[0]
    low, high = float(env.action_space.low[0]), float(env.action_space.high[0])

    print(f"Run directory: {outdir}")
    print(f"Genes: {num_genes}, Range: [{low}, {high}] A")

    def fitness_func(ga, solution, idx):
        env.reset()
        _, reward, _, _, _ = env.step(np.array(solution, dtype=np.float64))
        return float(reward)

    def on_parents(ga, parents):
        fitness = np.array(ga.last_generation_fitness)
        indices = np.array(ga.last_generation_parents_indices)
        ranking = np.argsort(fitness)[::-1]

        print(f"\n--- Parent Selection: Generation {ga.generations_completed + 1} ---")
        print("Population fitness:", fitness)
        print("Best indices:", ranking[:args.num_parents_mating])
        print("Best fitness:", fitness[ranking[:args.num_parents_mating]])
        print("Selected indices:", indices)
        print("Selected fitness:", fitness[indices])
        print("Selected parents:\n", parents)

    best_history = []
    previous_best = -np.inf

    def on_generation(ga):
        nonlocal previous_best

        gen = ga.generations_completed
        solution, fitness, idx = ga.best_solution()

        print(f"\nGeneration {gen}: Best fitness={fitness:.8f}, index={idx}")

        ga.save(os.path.join(outdir, "ga_instance"))

        np.savez(
            os.path.join(outdir, "results.npz"),
            best_solution=solution,
            best_fitness=fitness,
            best_solution_idx=idx,
            fitness_history=np.array(ga.best_solutions_fitness),
            bpm_names=np.array(env.unwrapped.bpm_names),
            control_names=np.array(env.unwrapped.control_names),
        )

        if fitness > previous_best:
            previous_best = fitness
            best_history.append({
                "generation": gen,
                "fitness": float(fitness),
                "solution": np.asarray(solution).tolist(),
            })
            with open(os.path.join(outdir, "best_history.json"), "w") as f:
                json.dump(best_history, f, indent=2)

    ga = pygad.GA(
        num_generations=args.num_generations,
        num_parents_mating=args.num_parents_mating,
        sol_per_pop=args.sol_per_pop,
        num_genes=num_genes,
        fitness_func=fitness_func,
        gene_space={"low": low, "high": high},
        parent_selection_type="sss",
        random_seed=args.seed,
        on_parents=on_parents,
        on_generation=on_generation,
    )

    try:
        ga.run()

        solution, fitness, idx = ga.best_solution()
        print(f"\nFinal best fitness: {fitness:.8f}")
        print(f"Final best solution:\n{solution}")

        plt.plot(ga.best_solutions_fitness)
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness (Reward)")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "fitness_plot.png"), dpi=150)
        plt.close()

    finally:
        env.close()


if __name__ == "__main__":
    main()