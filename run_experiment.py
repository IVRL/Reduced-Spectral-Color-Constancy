"""
Single-experiment entry point for the CBC pipeline.

Loads or builds a SimpleCBC model for the given settings, evaluates it on
the synthetic KAUST test set or the real radiance dataset, and optionally
saves the model and results to disk.
"""

from tqdm import tqdm

from experiment_store import CBCExperimentStore, RADIANCE_SCHEMA
from create_models import create_model
from experiment_settings import validate_kaust_settings
from test_model import test_model
from radiance.test_model import test_model_on_radiance


def run_experiment(
        settings: dict | None = None,
        save_model: bool = True,
        save_results: bool = True,
        test_on_radiance: bool = False,
) -> dict:
    """
    Build (or reload) a CBC model and evaluate it.

    If a saved model matching the given settings already exists in the result
    store, it is loaded from disk instead of being rebuilt.

    Args:
        settings:         Experiment settings dict.
        save_model:       Whether to save a newly built model to disk.
        save_results:     Whether to append metrics to the CSV result store.
        test_on_radiance: If True, evaluate on the real radiance dataset
                          instead of the synthetic KAUST test set.

    Returns:
        Metrics dict as returned by test_model() or test_model_on_radiance().
    """
    settings = validate_kaust_settings(settings)
    test_illuminant_names = settings["test_illuminant_names"]
    threshold = settings["threshold"]
    snr_db    = settings["snr_db"]

    store = CBCExperimentStore()
    model = store.get_model(settings)
    if not model:
        tqdm.write("Creating a new model")
        model = create_model(settings, verbose=True, tqdm_write=True)
    else:
        tqdm.write("Loaded a premade model that fit the settings provided")
        save_model = False

    tqdm.write(f'Combined histograms shape: {model.combined_hists.shape}')
    tqdm.write(f"{model.combined_hists.nbytes / 1e9:.2f} GB")

    if save_model:
        model_path = model.save()
        tqdm.write(f"Saved model at {model_path}")

    if test_on_radiance:
        res = test_model_on_radiance(model, test_illuminant_names, threshold=threshold)

        hdr = "Evaluation — Real radiance test set"
        rows = [
            ("Illuminant accuracy",     f"{res['accuracy']:.2%}"),
            ("mean_ang_err",            f"{res['mean_ang_err']:.4f}"),
            ("mean_ang_err_wp",         f"{res['mean_ang_err_wp']:.4f}"),
            ("mean_snapped_ang_err",    f"{res['mean_snapped_ang_err']:.4f}"),
            ("mean_snapped_ang_err_wp", f"{res['mean_snapped_ang_err_wp']:.4f}"),
        ]
        if save_results:
            radiance_store = CBCExperimentStore(
                file_name='cbc_radiance_runs.csv',
                schema=RADIANCE_SCHEMA,
            )
            radiance_store.add_run(settings=settings, metrics=res, model_path=model.path)
    else:
        res = test_model(model, test_illuminant_names, threshold=threshold, snr_db=snr_db)

        hdr = "Evaluation — Test set"
        rows = [
            ("Illuminant accuracy", f"{res['accuracy']:.2%}"),
            ("mean_ang_err",        f"{res['mean_ang_err']:.4f}"),
            ("mean_ang_err_wp",     f"{res['mean_ang_err_wp']:.4f}"),
        ]
        if save_results:
            store.add_run(settings=settings, metrics=res, model_path=model.path)

    w = max(len(k) for k, _ in rows)
    tqdm.write(f"\n{hdr}\n" + "\n".join(f"  {k:<{w}} : {v}" for k, v in rows) + "\n")

    return res
