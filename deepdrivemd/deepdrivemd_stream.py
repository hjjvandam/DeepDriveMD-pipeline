import os
import glob
import sys
import shutil
import argparse
import itertools
from typing import List
import radical.utils as ru
from radical.entk import AppManager, Pipeline, Stage, Task
from deepdrivemd.config_stream import ExperimentConfig, BaseStageConfig
from deepdrivemd.data.api import DeepDriveMD_API


def generate_task(cfg: BaseStageConfig) -> Task:
    task = Task()
    task.cpu_reqs = cfg.cpu_reqs.dict().copy()
    task.gpu_reqs = cfg.gpu_reqs.dict().copy()
    task.pre_exec = cfg.pre_exec.copy()
    task.executable = cfg.executable
    task.arguments = cfg.arguments.copy()
    return task


class PipelineManager:
    MOLECULAR_DYNAMICS_STAGE_NAME = "MolecularDynamics"
    AGGREGATION_STAGE_NAME = "Aggregating"
    MACHINE_LEARNING_STAGE_NAME = "MachineLearning"
    AGENT_STAGE_NAME = "Agent"

    MOLECULAR_DYNAMICS_PIPELINE_NAME = "MolecularDynamics_pipeline"
    AGGREGATION_PIPELINE_NAME = "Aggregating_pipeline"
    MACHINE_LEARNING_PIPELINE_NAME = "MachineLearning_pipeline"
    AGENT_PIPELINE_NAME = "Agent_pipeline"
    
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg

        '''
        #### for debugging
        print('='*20)
        print(type(cfg))
        print(dir(cfg))
        print("="*3)
        print(cfg)
        print('='*20)
        sys.stdout.flush()
        ###
        '''

        self.stage_idx = 0 ###?
        
        self.api = DeepDriveMD_API(cfg.experiment_directory) ###?

        '''
        ### for debugging
        print('='*20)
        print(type(self.api))
        print(dir(self.api))
        print("="*3)
        print(self.api)
        print('='*20)
        sys.stdout.flush()
        ### 
        '''
        
        self.pipelines = {}

        p_md = Pipeline()
        p_md.name = self.MOLECULAR_DYNAMICS_PIPELINE_NAME
        
        self.pipelines[p_md.name] = p_md
        
        p_aggregate = Pipeline()
        p_aggregate.name = self.AGGREGATION_PIPELINE_NAME

        self.pipelines[p_aggregate.name] = p_aggregate


        '''
        p_aggregate = []
        for k in range(cfg.aggregation_stage.num_tasks): 
            p_aggregate.append(Pipeline())
            p_aggregate[k].name = 'aggregate_pipeline_%d' % k
            self.pipelines[p_aggregate[k].name] = p_aggregate[k]
           
        '''

        p_ml = Pipeline()
        p_ml.name = self.MACHINE_LEARNING_PIPELINE_NAME
        self.pipelines[p_ml.name] = p_ml
            
        p_outliers = Pipeline()
        p_outliers.name = self.AGENT_PIPELINE_NAME
        self.pipelines[p_outliers.name] = p_outliers
            
        self._init_experiment_dir()

    def _init_experiment_dir(self):
        # Make experiment directories
        self.cfg.experiment_directory.mkdir()
        self.api.molecular_dynamics_stage.runs_dir.mkdir()
        self.api.aggregation_stage.runs_dir.mkdir()
        self.api.machine_learning_stage.runs_dir.mkdir()
        # self.api.model_selection_stage.runs_dir.mkdir()
        self.api.agent_stage.runs_dir.mkdir()


    def _generate_pipeline_iteration(self):

        self.pipeline.add_stages(self.generate_molecular_dynamics_stage())

        if not cfg.aggregation_stage.skip_aggregation:
            self.pipeline.add_stages(self.generate_aggregating_stage())

        if self.stage_idx % cfg.machine_learning_stage.retrain_freq == 0:
            self.pipeline.add_stages(self.generate_machine_learning_stage())
        self.pipeline.add_stages(self.generate_model_selection_stage())

        agent_stage = self.generate_agent_stage()
        agent_stage.post_exec = self.func_condition
        self.pipeline.add_stages(agent_stage)

        self.stage_idx += 1

    def generate_pipelines(self) -> List[Pipeline]:
        # self._generate_pipeline_iteration()
        return self.pipelines

    def generate_molecular_dynamics_pipeline(self):
        self.pipelines

    def generate_molecular_dynamics_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.MOLECULAR_DYNAMICS_STAGE_NAME
        cfg = self.cfg.molecular_dynamics_stage
        stage_api = self.api.molecular_dynamics_stage


        ### for debugging
        print('='*20)
        print(type(cfg))
        print(dir(cfg))
        print("="*3)
        print(cfg)
        print('='*20)
        sys.stdout.flush()
        ### 


        ### for debugging
        print('='*20)
        print(type(self.api))
        print(dir(self.api))
        print("="*3)
        print(self.api)
        print('='*20)
        sys.stdout.flush()
        ### 

        '''
        if self.stage_idx == 0:
            filenames = self.api.get_initial_pdbs(cfg.task_config.initial_pdb_dir)
            filenames = itertools.cycle(filenames)
        else:
            filenames = None
        '''

        init_file = glob.glob(str(cfg.task_config.initial_pdb_dir) +"/*.pdb")[0]

        for task_idx in range(cfg.num_tasks):

            output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
            assert output_path is not None

            # Update base parameters
            cfg.task_config.experiment_directory = self.cfg.experiment_directory
            cfg.task_config.stage_idx = self.stage_idx
            cfg.task_config.task_idx = task_idx
            cfg.task_config.node_local_path = self.cfg.node_local_path
            cfg.task_config.output_path = output_path
            cfg.task_config.pdb_file = init_file

            cfg_path = stage_api.config_path(self.stage_idx, task_idx)
            cfg.task_config.dump_yaml(cfg_path)
            task = generate_task(cfg)
            task.arguments += ["-c", cfg_path.as_posix()]
            stage.add_tasks(task)

        return stage

    def generate_aggregating_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.AGGREGATION_STAGE_NAME
        cfg = self.cfg.aggregation_stage
        stage_api = self.api.aggregation_stage

        task_idx = 0
        output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
        assert output_path is not None

        # Update base parameters
        cfg.task_config.experiment_directory = self.cfg.experiment_directory
        cfg.task_config.stage_idx = self.stage_idx
        cfg.task_config.task_idx = task_idx
        cfg.task_config.node_local_path = self.cfg.node_local_path
        cfg.task_config.output_path = output_path

        # Write yaml configuration
        cfg_path = stage_api.config_path(self.stage_idx, task_idx)
        cfg.task_config.dump_yaml(cfg_path)
        task = generate_task(cfg)
        task.arguments += ["-c", cfg_path.as_posix()]
        stage.add_tasks(task)

        return stage

    def generate_machine_learning_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.MACHINE_LEARNING_STAGE_NAME
        cfg = self.cfg.machine_learning_stage
        stage_api = self.api.machine_learning_stage

        task_idx = 0
        output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
        assert output_path is not None

        # Update base parameters
        cfg.task_config.experiment_directory = self.cfg.experiment_directory
        cfg.task_config.stage_idx = self.stage_idx
        cfg.task_config.task_idx = task_idx
        cfg.task_config.node_local_path = self.cfg.node_local_path
        cfg.task_config.output_path = output_path
        cfg.task_config.model_tag = stage_api.unique_name(output_path)
        if self.stage_idx > 0:
            # Machine learning should use model selection API
            cfg.task_config.init_weights_path = None

        # Write yaml configuration
        cfg_path = stage_api.config_path(self.stage_idx, task_idx)
        cfg.task_config.dump_yaml(cfg_path)
        task = generate_task(cfg)
        task.arguments += ["-c", cfg_path.as_posix()]
        stage.add_tasks(task)

        return stage

    def generate_model_selection_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.MODEL_SELECTION_STAGE_NAME
        cfg = self.cfg.model_selection_stage
        stage_api = self.api.model_selection_stage

        task_idx = 0
        output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
        assert output_path is not None

        # Update base parameters
        cfg.task_config.experiment_directory = self.cfg.experiment_directory
        cfg.task_config.stage_idx = self.stage_idx
        cfg.task_config.task_idx = task_idx
        cfg.task_config.node_local_path = self.cfg.node_local_path
        cfg.task_config.output_path = output_path

        # Write yaml configuration
        cfg_path = stage_api.config_path(self.stage_idx, task_idx)
        cfg.task_config.dump_yaml(cfg_path)
        task = generate_task(cfg)
        task.arguments += ["-c", cfg_path.as_posix()]
        stage.add_tasks(task)

        return stage

    def generate_agent_stage(self) -> Stage:
        stage = Stage()
        stage.name = self.AGENT_STAGE_NAME
        cfg = self.cfg.agent_stage
        stage_api = self.api.agent_stage

        task_idx = 0
        output_path = stage_api.task_dir(self.stage_idx, task_idx, mkdir=True)
        assert output_path is not None

        # Update base parameters
        cfg.task_config.experiment_directory = self.cfg.experiment_directory
        cfg.task_config.stage_idx = self.stage_idx
        cfg.task_config.task_idx = task_idx
        cfg.task_config.node_local_path = self.cfg.node_local_path
        cfg.task_config.output_path = output_path

        # Write yaml configuration
        cfg_path = stage_api.config_path(self.stage_idx, task_idx)
        cfg.task_config.dump_yaml(cfg_path)
        task = generate_task(cfg)
        task.arguments += ["-c", cfg_path.as_posix()]
        stage.add_tasks(task)

        return stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", help="YAML config file", type=str, required=True
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)

    pipeline_manager = PipelineManager(cfg)

    pipeline_manager.generate_molecular_dynamics_stage()


    sys.exit(0)

    reporter = ru.Reporter(name="radical.entk")
    reporter.title(cfg.title)

    # Create Application Manager
    try:
        appman = AppManager(
            hostname=os.environ["RMQ_HOSTNAME"],
            port=int(os.environ["RMQ_PORT"]),
            username=os.environ["RMQ_USERNAME"],
            password=os.environ["RMQ_PASSWORD"],
        )
    except KeyError:
        raise ValueError(
            "Invalid RMQ environment. Please see README.md for configuring environment."
        )

    # Calculate total number of nodes required. Assumes 1 MD job per GPU
    # TODO: fix this assumption for NAMD
    num_full_nodes, extra_gpus = divmod(
        cfg.molecular_dynamics_stage.num_tasks, cfg.gpus_per_node
    )
    extra_node = int(extra_gpus > 0)
    num_nodes = max(1, num_full_nodes + extra_node)

    appman.resource_desc = {
        "resource": cfg.resource,
        "queue": cfg.queue,
        "schema": cfg.schema_,
        "walltime": cfg.walltime_min,
        "project": cfg.project,
        "cpus": cfg.cpus_per_node * cfg.hardware_threads_per_cpu * num_nodes,
        "gpus": cfg.gpus_per_node * num_nodes,
    }

    pipeline_manager = PipelineManager(cfg)
    # Back up configuration file (PipelineManager must create cfg.experiment_dir)
    shutil.copy(args.config, cfg.experiment_directory)

    pipelines = pipeline_manager.generate_pipelines()
    # Assign the workflow as a list of Pipelines to the Application Manager.
    # All the pipelines in the list will execute concurrently.
    appman.workflow = pipelines

    # Run the Application Manager
    appman.run()
