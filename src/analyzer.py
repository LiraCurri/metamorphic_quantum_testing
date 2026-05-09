import time
import pandas as pd
import datetime


class ResultAnalyzer:
    """
    Records, summarizes, and exports results from metamorphic testing experiments.

    Supports:
    - Circuit-level experiments (distribution distance, violations)
    - VQE experiments (energy difference, trace metrics)

    """

    def __init__(self):
        self.records = []
        self.df = pd.DataFrame()
        self.start_time = time.time()

    # CIRCUIT RECORDING
    def record(
        self,
        test_id: int,
        relation_name: str,
        distance: float,
        violation: bool,
        runtime: float | None = None,
    ):
        """Store a single circuit experiment result."""
        entry = {
            "test_id": test_id,
            "relation": relation_name,
            "distance": distance,
            "violation": bool(violation),
            "runtime": runtime,
        }

        self.records.append(entry)
        self.df = pd.DataFrame(self.records)

    # VQE RECORDING
    def record_vqe(
        self,
        test_id: int,
        relation_name: str,
        delta_E: float,
        violation: bool,
        avg_energy_diff: float | None = None,
        max_energy_diff: float | None = None,
        avg_sym_diff: float | None = None,
        max_sym_diff: float | None = None,
        avg_param_diff: float | None = None,
        max_param_diff: float | None = None,
        runtime: float | None = None,
    ):
        """Store a VQE experiment result with extended metrics."""
        entry = {
            "test_id": test_id,
            "relation": relation_name,
            "delta_E": delta_E,
            "violation": bool(violation),
            "avg_energy_diff": avg_energy_diff,
            "max_energy_diff": max_energy_diff,
            "avg_sym_diff": avg_sym_diff,
            "max_sym_diff": max_sym_diff,
            "avg_param_diff": avg_param_diff,
            "max_param_diff": max_param_diff,
            "runtime": runtime,
        }

        self.records.append(entry)
        self.df = pd.DataFrame(self.records)

    # SUMMARY
    def summary(self):
        """Print high-level experiment statistics."""
        if self.df.empty:
            print("No results recorded.")
            return

        total_tests = len(self.df)
        violations = int(self.df["violation"].sum())
        passed = total_tests - violations
        violation_rate = violations / total_tests

        print("\n============================")
        print("Experiment Summary")
        print("============================")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed}")
        print(f"Violations: {violations}")
        print(f"Violation rate: {violation_rate:.4f}")

        # Circuit metric
        if "distance" in self.df.columns:
            avg_distance = self.df["distance"].mean()
            print(f"Average distribution distance: {avg_distance:.6f}")

        # VQE metric
        if "delta_E" in self.df.columns:
            avg_delta_E = self.df["delta_E"].mean()
            print(f"Average energy difference (ΔE): {avg_delta_E:.6f}")

        # Runtime
        if "runtime" in self.df.columns and self.df["runtime"].notnull().any():
            avg_runtime = self.df["runtime"].mean()
            print(f"Average runtime per test: {avg_runtime:.4f} sec")
            print(f"Total experiment runtime: {self.runtime():.2f} sec")

        print("============================\n")

    # RELATION BREAKDOWN
    def relation_breakdown(self):
        """Show statistics grouped by metamorphic relation."""
        if self.df.empty:
            print("No results recorded.")
            return

        grouped = self.df.groupby("relation")

        print("\nRelation Breakdown")
        print("------------------")

        for relation, group in grouped:
            total = len(group)
            violations = int(group["violation"].sum())
            rate = violations / total

            print(f"{relation}")
            print(f"  tests: {total}")
            print(f"  violations: {violations}")
            print(f"  violation rate: {rate:.4f}")

            # Circuit metric
            if "distance" in group.columns:
                avg_dist = group["distance"].mean()
                print(f"  avg distance: {avg_dist:.6f}")

            # VQE metric
            if "delta_E" in group.columns:
                avg_delta_E = group["delta_E"].mean()
                print(f"  avg ΔE: {avg_delta_E:.6f}")

            print()

    # EXPORT
    def export_csv(self, filename: str | None = None):
        """Save experiment data to CSV (timestamped by default)."""
        if self.df.empty:
            print("No data to export.")
            return

        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/experiment_results_{timestamp}.csv"

        self.df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")

    # RUNTIME
    def runtime(self):
        """Return total experiment runtime."""
        return time.time() - self.start_time
