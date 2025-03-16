import os

class WandbLogger:
    """
    Log using `Weights and Biases` for depth estimation.
    """
    def __init__(self, opt):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "To use the Weights and Biases Logger please install wandb."
                "Run `pip install wandb` to install it."
            )

        self._wandb = wandb

        # ✅ Initialize a W&B run
        if self._wandb.run is None:
            self._wandb.init(
                project=opt['wandb']['project'],
                config=opt,
                dir='./experiments'
            )

        self.config = self._wandb.config

        # ✅ Updated table for MAE and RMSE (depth estimation metrics)
        if self.config.get('log_eval', None):
            self.eval_table = self._wandb.Table(columns=['predicted_depth', 
                                                         'ground_truth_depth', 
                                                         'mae', 
                                                         'rmse'])
        else:
            self.eval_table = None

        if self.config.get('log_infer', None):
            self.infer_table = self._wandb.Table(columns=['predicted_depth', 
                                                          'ground_truth_depth'])
        else:
            self.infer_table = None

    def log_metrics(self, metrics, commit=True): 
        """
        Log train/validation metrics onto W&B.
        """
        self._wandb.log(metrics, commit=commit)

    def log_image(self, key_name, depth_map):
        """
        Log depth map array onto W&B.
        """
        self._wandb.log({key_name: self._wandb.Image(depth_map)})

    def log_images(self, key_name, list_depth_maps):
        """
        Log list of depth map arrays onto W&B
        """
        self._wandb.log({key_name: [self._wandb.Image(depth) for depth in list_depth_maps]})

    # ✅ Updated to reflect depth estimation checkpoints
    def log_checkpoint(self, current_epoch, current_step):
        """
        Log the model checkpoint as W&B artifacts
        """
        model_artifact = self._wandb.Artifact(
            self._wandb.run.id + "_model", type="model"
        )

        gen_path = os.path.join(
            self.config.path['checkpoint'], f'I{current_step}_E{current_epoch}_depth_gen.pth')
        opt_path = os.path.join(
            self.config.path['checkpoint'], f'I{current_step}_E{current_epoch}_depth_opt.pth')

        model_artifact.add_file(gen_path)
        model_artifact.add_file(opt_path)
        self._wandb.log_artifact(model_artifact, aliases=["latest"])

    # ✅ Updated to handle MAE and RMSE for depth evaluation
    def log_eval_data(self, pred_depth, gt_depth, mae=None, rmse=None):
        """
        Add data row-wise to the initialized table.
        """
        if mae is not None and rmse is not None:
            self.eval_table.add_data(
                self._wandb.Image(pred_depth),
                self._wandb.Image(gt_depth),
                mae,
                rmse
            )
        else:
            self.infer_table.add_data(
                self._wandb.Image(pred_depth),
                self._wandb.Image(gt_depth)
            )

    def log_eval_table(self, commit=False):
        """
        Log the table
        """
        if self.eval_table:
            self._wandb.log({'eval_data': self.eval_table}, commit=commit)
        elif self.infer_table:
            self._wandb.log({'infer_data': self.infer_table}, commit=commit)
