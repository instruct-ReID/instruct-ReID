from .data_builder_attr import DataBuilder_attr
from .data_builder_sc import DataBuilder_sc
from .data_builder_cc import DataBuilder_cc
from .data_builder_ctcc import DataBuilder_ctcc


def dataset_entry(this_task_info):
    return globals()[this_task_info.task_name]