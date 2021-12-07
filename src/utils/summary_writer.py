import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter as OriginalSummaryWriter
from torch.utils.tensorboard.summary import hparams


##########################################################################
# Since the desired behavior could not be achieved with the original     #
# SummaryWriter from torch.utils.tensorboard (torch==1.9.0) the class is #
# modified here. The code is unchanged except of the commented part      #
##########################################################################
class SummaryWriter(OriginalSummaryWriter):
    # The original add_hparams function did not function as necessary,
    # therefore a slight adaption was made to the function
    def add_hparams(
        self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None
    ):
        """Add a set of hyperparameters to be compared in TensorBoard.

        Args:
            hparam_dict (dict): Each key-value pair in the dictionary is the
              name of the hyper parameter and it's corresponding value.
              The type of the value can be one of `bool`, `string`, `float`,
              `int`, or `None`.
            metric_dict (dict): Each key-value pair in the dictionary is the
              name of the metric and it's corresponding value. Note that the key used
              here should be unique in the tensorboard record. Otherwise the value
              you added by ``add_scalar`` will be displayed in hparam plugin. In most
              cases, this is unwanted.
            hparam_domain_discrete: (Optional[Dict[str, List[Any]]]) A dictionary that
              contains names of the hyperparameters and all discrete values they can hold
            run_name (str): Name of the run, to be included as part of the logdir.
              If unspecified, will use current timestamp.

        Examples::

            from torch.utils.tensorboard import SummaryWriter
            with SummaryWriter() as w:
                for i in range(5):
                    w.add_hparams({'lr': 0.1*i, 'bsize': i},
                                  {'hparam/accuracy': 10*i, 'hparam/loss': 10*i})

        Expected result:

        .. image:: _static/img/tensorboard/add_hparam.png
           :scale: 50 %

        """
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        ### Modified part ###
        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            if v is not None:
                self.add_scalar(k, v)
        #####################
