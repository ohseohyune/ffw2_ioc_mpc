"""
Episode trajectory sanity checker.

Usage:
    python -m ffw2_ioc_mpc.utils.check_episode_data \
      --data_dir data/raw/episode_xxx \
      --system_params configs/system_params.yaml \
      --cost_params configs/cost_params.yaml
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import yaml


REQUIRED_FILES = {
    "qpos": "qpos_traj.npy",
    "qvel": "qvel_traj.npy",
    "input": "input_traj.npy",
    "M": "M_traj.npy",
    "CG": "CG_traj.npy",
    "ys": "ys_traj.npy",
}


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_episode_data(data_dir: str) -> Dict[str, np.ndarray]:
    data: Dict[str, np.ndarray] = {}
    for key, fname in REQUIRED_FILES.items():
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing file: {fpath}")
        data[key] = np.load(fpath)
    return data


def finite_stats(arr: np.ndarray) -> Tuple[int, int]:
    finite_mask = np.isfinite(arr)
    n_total = int(arr.size)
    n_bad = int(n_total - finite_mask.sum())
    return n_total, n_bad


def summarize_array(name: str, arr: np.ndarray) -> str:
    flat = arr.reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return f"{name:<5} all non-finite"
    return (
        f"{name:<5} shape={arr.shape} "
        f"min={float(finite.min()): .3e} max={float(finite.max()): .3e} "
        f"mean={float(finite.mean()): .3e} std={float(finite.std()): .3e}"
    )


def check_shapes(
    data: Dict[str, np.ndarray],
    system_params: dict | None,
    cost_params: dict | None,
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    qpos = data["qpos"]
    qvel = data["qvel"]
    inp = data["input"]
    M = data["M"]
    CG = data["CG"]
    ys = data["ys"]

    first_dims = {k: int(v.shape[0]) for k, v in data.items()}
    if len(set(first_dims.values())) != 1:
        errors.append(f"Inconsistent first dimension N across arrays: {first_dims}")
        return errors, warnings

    N = qpos.shape[0]

    if qpos.ndim != 2:
        errors.append(f"qpos must be 2D, got {qpos.shape}")
    if qvel.ndim != 2:
        errors.append(f"qvel must be 2D, got {qvel.shape}")
    if inp.ndim != 2:
        errors.append(f"input must be 2D, got {inp.shape}")
    if ys.ndim != 2:
        errors.append(f"ys must be 2D, got {ys.shape}")
    if M.ndim != 3:
        errors.append(f"M must be 3D, got {M.shape}")
    if CG.ndim != 2:
        errors.append(f"CG must be 2D, got {CG.shape}")

    if not errors:
        nq = qpos.shape[1]
        nv = qvel.shape[1]
        nu_raw = inp.shape[1]
        target_dim = ys.shape[1]

        if M.shape != (N, nv, nv):
            errors.append(f"M shape {M.shape} != (N={N}, nv={nv}, nv={nv})")
        if CG.shape != (N, nv):
            errors.append(f"CG shape {CG.shape} != (N={N}, nv={nv})")

        if nq != nv + 1:
            warnings.append(
                f"qpos/qvel dim relation unusual: nq={nq}, nv={nv} (expected nq≈nv+1 for current robot)"
            )

        if system_params:
            sys_cfg = system_params.get("system", {})
            exp_nu = sys_cfg.get("input_dimension")
            if exp_nu is not None and int(exp_nu) != nu_raw:
                warnings.append(
                    f"input_traj dim={nu_raw} differs from system.input_dimension={exp_nu} "
                    "(can be OK if raw input saves full actuators but MPC uses subset)"
                )

        if cost_params:
            idxs = (
                cost_params.get("cost_function", {})
                .get("target_state_selection", {})
                .get("indices", [])
            )
            if idxs and len(idxs) != target_dim:
                warnings.append(
                    f"ys target_dim={target_dim} but cost target_state_selection.indices has {len(idxs)} entries"
                )

    return errors, warnings


def check_finite(data: Dict[str, np.ndarray]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    for name, arr in data.items():
        n_total, n_bad = finite_stats(arr)
        if n_bad > 0:
            errors.append(f"{name}: non-finite values {n_bad}/{n_total}")
    return errors, warnings


def check_variation(data: Dict[str, np.ndarray]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    for name in ["qpos", "qvel", "input", "CG", "ys"]:
        arr = data[name]
        if arr.ndim < 2:
            continue
        arr2 = arr.reshape(arr.shape[0], -1)
        per_col_std = np.std(arr2, axis=0)
        zero_var_cols = np.where(per_col_std < 1e-12)[0]
        if zero_var_cols.size == arr2.shape[1]:
            warnings.append(f"{name}: all columns constant (std≈0)")
        elif zero_var_cols.size > 0:
            warnings.append(
                f"{name}: {zero_var_cols.size}/{arr2.shape[1]} columns constant (std≈0)"
            )

        if arr2.shape[0] >= 2:
            d = np.diff(arr2, axis=0)
            step_norm = np.linalg.norm(d, axis=1)
            if np.all(step_norm < 1e-12):
                warnings.append(f"{name}: no temporal change across steps")

    ys = data["ys"]
    if np.allclose(ys, 0.0):
        warnings.append("ys: all zeros (likely dummy target)")
    elif np.all(np.std(ys, axis=0) < 1e-12):
        warnings.append("ys: constant target across entire episode")

    return errors, warnings


def check_matrix_properties(data: Dict[str, np.ndarray]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    M = data["M"]
    if M.ndim != 3:
        return errors, warnings

    sym_err = np.max(np.abs(M - np.transpose(M, (0, 2, 1))))
    if not np.isfinite(sym_err):
        errors.append("M: symmetry check produced non-finite value")
    elif sym_err > 1e-6:
        warnings.append(f"M: max symmetry error {float(sym_err):.3e} (>1e-6)")

    diag = np.diagonal(M, axis1=1, axis2=2)
    min_diag = float(np.min(diag))
    if min_diag <= 0.0:
        warnings.append(f"M: non-positive diagonal detected (min diag={min_diag:.3e})")

    # Sample eigenvalue checks to keep runtime bounded.
    n = M.shape[0]
    sample_idx = np.unique(np.linspace(0, n - 1, min(n, 10), dtype=int))
    min_eigs = []
    for i in sample_idx:
        try:
            evals = np.linalg.eigvalsh((M[i] + M[i].T) * 0.5)
            min_eigs.append(float(evals.min()))
        except np.linalg.LinAlgError:
            warnings.append(f"M[{i}]: eigvalsh failed")
    if min_eigs:
        if min(min_eigs) <= 0.0:
            warnings.append(
                f"M: sampled min eigenvalue <= 0 (min={min(min_eigs):.3e}); mass matrix may be invalid"
            )

    return errors, warnings


def _get_target_qpos_indices(cost_params: dict | None) -> List[int]:
    if not cost_params:
        return []
    idxs = (
        cost_params.get("cost_function", {})
        .get("target_state_selection", {})
        .get("indices", [])
    )
    if not isinstance(idxs, list):
        return []
    return [int(i) for i in idxs]


def plot_qpos_vs_ys(
    data: Dict[str, np.ndarray],
    cost_params: dict | None,
    data_dir: str,
    save_path: str | None = None,
    no_show: bool = False,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as ex:
        print(f"[Plot] matplotlib import 실패: {ex}")
        return

    qpos = data.get("qpos")
    ys = data.get("ys")
    if qpos is None or ys is None:
        print("[Plot] qpos 또는 ys 데이터가 없어 그래프를 그릴 수 없습니다.")
        return
    if qpos.ndim != 2 or ys.ndim != 2:
        print(f"[Plot] qpos/ys ndim 오류: qpos{qpos.shape}, ys{ys.shape}")
        return

    idxs = _get_target_qpos_indices(cost_params)
    if idxs and len(idxs) == ys.shape[1]:
        if max(idxs) >= qpos.shape[1] or min(idxs) < 0:
            print(
                f"[Plot] cost target indices out of range for qpos shape {qpos.shape}: {idxs}\n"
                "       → qpos 첫 ys_dim 열과 비교합니다."
            )
            idxs = list(range(min(qpos.shape[1], ys.shape[1])))
    else:
        if idxs:
            print(
                f"[Plot] indices 길이({len(idxs)})와 ys dim({ys.shape[1]})가 달라서 "
                "qpos 첫 ys_dim 열과 비교합니다."
            )
        idxs = list(range(min(qpos.shape[1], ys.shape[1])))

    if len(idxs) != ys.shape[1]:
        print(
            f"[Plot] 비교 차원 불일치: selected qpos dims={len(idxs)}, ys dim={ys.shape[1]} "
            "→ plotting 생략"
        )
        return

    q_sel = qpos[:, idxs]
    err = q_sel - ys
    n_dim = ys.shape[1]
    t = np.arange(qpos.shape[0])

    # 차원이 많아도 읽기 좋게 최대 4열로 배치
    ncols = min(4, n_dim) if n_dim > 0 else 1
    nrows = int(np.ceil(n_dim / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 2.8 * nrows),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for d in range(n_dim):
        ax = axes_flat[d]
        ax.plot(t, q_sel[:, d], label=f"qpos[{idxs[d]}]", linewidth=1.3)
        ax.plot(t, ys[:, d], label=f"ys[{d}]", linewidth=1.1, alpha=0.9)
        ax.set_title(f"Dim {d}  (qpos[{idxs[d]}])")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    for d in range(n_dim, len(axes_flat)):
        axes_flat[d].axis("off")

    fig.suptitle(f"qpos(target indices) vs ys  |  {os.path.basename(data_dir)}", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    fig_err, axes_err = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.2 * ncols, 2.6 * nrows),
        sharex=True,
        squeeze=False,
    )
    axes_err_flat = axes_err.flatten()
    rmse_per_dim = np.sqrt(np.mean(err * err, axis=0))

    for d in range(n_dim):
        ax = axes_err_flat[d]
        ax.plot(t, err[:, d], linewidth=1.1)
        ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.5)
        ax.set_title(f"Error dim {d}  RMSE={rmse_per_dim[d]:.4f}")
        ax.grid(True, alpha=0.3)

    for d in range(n_dim, len(axes_err_flat)):
        axes_err_flat[d].axis("off")

    fig_err.suptitle("Tracking Error: qpos(target) - ys", fontsize=12)
    fig_err.tight_layout(rect=[0, 0.02, 1, 0.95])

    print(
        "[Plot] tracking RMSE:",
        float(np.sqrt(np.mean(err * err))),
        "| per-dim:",
        np.round(rmse_per_dim, 6),
    )

    if save_path:
        root, ext = os.path.splitext(save_path)
        if not ext:
            ext = ".png"
            save_path = root + ext
        err_save = f"{root}_error{ext}"
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
        fig_err.savefig(err_save, dpi=140, bbox_inches="tight")
        print(f"[Plot] 저장: {save_path}")
        print(f"[Plot] 저장: {err_save}")

    if no_show:
        plt.close(fig)
        plt.close(fig_err)
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity check episode trajectory npy files")
    p.add_argument("--data_dir", required=True, help="Episode directory containing *_traj.npy")
    p.add_argument("--system_params", default="configs/system_params.yaml")
    p.add_argument("--cost_params", default="configs/cost_params.yaml")
    p.add_argument(
        "--plot_tracking",
        action="store_true",
        help="qpos(target indices)와 ys_traj를 겹쳐서 그리고 오차 그래프도 표시",
    )
    p.add_argument(
        "--plot_save_path",
        default=None,
        help="그래프 저장 경로 (예: tmp/tracking.png). 지정 시 error 그래프는 *_error.png로 저장",
    )
    p.add_argument(
        "--no_show",
        action="store_true",
        help="그래프 창을 띄우지 않고 저장만 수행 (--plot_tracking와 함께 사용)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = os.path.abspath(args.data_dir)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    system_params = load_yaml(args.system_params) if os.path.exists(args.system_params) else None
    cost_params = load_yaml(args.cost_params) if os.path.exists(args.cost_params) else None

    data = load_episode_data(data_dir)

    errors: List[str] = []
    warnings: List[str] = []

    for fn in (check_shapes, check_finite, check_variation, check_matrix_properties):
        if fn is check_shapes:
            e, w = fn(data, system_params, cost_params)
        else:
            e, w = fn(data)
        errors.extend(e)
        warnings.extend(w)

    print("=" * 70)
    print("Episode Data Sanity Check")
    print("=" * 70)
    print(f"data_dir: {data_dir}")
    print("")
    print("[Summary]")
    for name in ["qpos", "qvel", "input", "M", "CG", "ys"]:
        print("  " + summarize_array(name, data[name]))

    if "ys" in data and data["ys"].ndim == 2:
        ys = data["ys"]
        print("")
        print("[ys detail]")
        print(f"  shape        : {ys.shape}")
        print(f"  unique_count : {np.unique(ys).size}")
        print(f"  first_rows   : {np.round(ys[:5], 6)}")

    print("")
    print("[Findings]")
    if not errors and not warnings:
        print("  PASS: no issues detected")
    else:
        for msg in errors:
            print(f"  ERROR: {msg}")
        for msg in warnings:
            print(f"  WARN : {msg}")

    if args.plot_tracking:
        print("")
        print("[Plot] qpos(target indices) vs ys")
        plot_qpos_vs_ys(
            data=data,
            cost_params=cost_params,
            data_dir=data_dir,
            save_path=args.plot_save_path,
            no_show=args.no_show,
        )

    print("")
    if errors:
        print("Result: FAIL")
        raise SystemExit(1)
    if warnings:
        print("Result: WARN")
        raise SystemExit(2)
    print("Result: PASS")


if __name__ == "__main__":
    main()
