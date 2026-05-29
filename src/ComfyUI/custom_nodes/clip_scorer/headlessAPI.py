from utils import paths
from utils.config_params import get_run_params2
from workflow_api import wf_transformer


def test_custom_node():
    cfg = paths.get_cfg()
    p = get_run_params2()
    test_json_name = p["test_json"]

    srv_cfg = cfg['server']
    test_endpoint = srv_cfg['test_endpoint']

    test_workflow = wf_transformer.load_workflow_by_filename(test_json_name)

    # prompt_id = send2srv(test_workflow, cfg, test_endpoint)

if __name__ == "__main__":
    test_custom_node()
