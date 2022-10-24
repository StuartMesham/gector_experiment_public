import json
import os
from transformers.utils import (
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    cached_path,
    hf_bucket_url,
    is_offline_mode,
    is_remote_url,
    logging,
)
from typing import Any, Dict, Union
from requests import HTTPError
logger = logging.get_logger(__name__)


def get_inf_tweaks_dict(pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> Dict[str, Any]:
    cache_dir = kwargs.pop("cache_dir", None)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    use_auth_token = kwargs.pop("use_auth_token", None)
    local_files_only = kwargs.pop("local_files_only", False)
    revision = kwargs.pop("revision", None)
    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)

    user_agent = {"file_type": "config", "from_auto_class": from_auto_class}
    if from_pipeline is not None:
        user_agent["using_pipeline"] = from_pipeline

    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    pretrained_model_name_or_path = str(pretrained_model_name_or_path)
    if os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
        inf_tweaks_file = pretrained_model_name_or_path
    else:
        inference_tweaks_file = 'inference_tweak_params.json'

        if os.path.isdir(pretrained_model_name_or_path):
            inf_tweaks_file = os.path.join(pretrained_model_name_or_path, inference_tweaks_file)
        else:
            inf_tweaks_file = hf_bucket_url(
                pretrained_model_name_or_path, filename=inference_tweaks_file, revision=revision, mirror=None
            )

    try:
        # Load from URL or cache if already cached
        resolved_config_file = cached_path(
            inf_tweaks_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            user_agent=user_agent,
        )

    except RepositoryNotFoundError:
        raise EnvironmentError(
            f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier listed on "
            "'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a token having "
            "permission to this repo with `use_auth_token` or log in with `huggingface-cli login` and pass "
            "`use_auth_token=True`."
        )
    except RevisionNotFoundError:
        raise EnvironmentError(
            f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this "
            f"model name. Check the model page at 'https://huggingface.co/{pretrained_model_name_or_path}' for "
            "available revisions."
        )
    # except EntryNotFoundError:
    #     raise EnvironmentError(
    #         f"{pretrained_model_name_or_path} does not appear to have a file named {inference_tweaks_file}."
    #     )
    except HTTPError as err:
        raise EnvironmentError(
            f"There was a specific connection error when trying to load {pretrained_model_name_or_path}:\n{err}"
        )
    except ValueError:
        raise EnvironmentError(
            f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it in"
            f" the cached files and it looks like {pretrained_model_name_or_path} is not the path to a directory"
            f" containing a {inference_tweaks_file} file.\nCheckout your internet connection or see how to run the"
            " library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
        )
    except EnvironmentError:
        raise EnvironmentError(
            f"Can't load config for '{pretrained_model_name_or_path}'. If you were trying to load it from "
            "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
            f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
            f"containing a {inference_tweaks_file} file"
        )

    try:
        # Load config dict
        with open(resolved_config_file, 'r') as f:
            inf_tweaks_dict = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise EnvironmentError(
            f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file."
        )

    if resolved_config_file == inf_tweaks_file:
        logger.info(f"loading inference tweak parameter file {inf_tweaks_file}")
    else:
        logger.info(f"loading inference tweak parameter file {inf_tweaks_file} from cache at {resolved_config_file}")

    return inf_tweaks_dict
