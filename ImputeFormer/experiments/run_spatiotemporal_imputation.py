import copy
import datetime
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from tsl import config, logger
from tsl.data import SpatioTemporalDataModule, ImputationDataset
from tsl.data.loader import StaticGraphLoader
from tsl.data.preprocessing import StandardScaler
from tsl.imputers import Imputer
from tsl.nn.metrics import MaskedMetric, MaskedMAE, MaskedMSE, MaskedMRE
from tsl.nn.utils import casting
from tsl.utils import parser_utils, numpy_metrics
from tsl.utils.parser_utils import ArgParser

from imputeformer.imputation_ops import add_missing_values
from imputeformer.baselines import SAITS, TransformerModel, BRITS, SPINModel
from tsl.nn.models.imputation import GRINModel
from imputeformer.imputers import SPINImputer, SAITSImputer, BRITSImputer
from imputeformer.models import ImputeFormerModel
from imputeformer.scheduler import CosineSchedulerWithRestarts
from imputeformer.imputers import ImputeFormerImputer
from imputeformer.datasets import AirQuality, MetrLA, PemsBay, PeMS03, PeMS04, PeMS07, PeMS08, SolarBenchmark, Elergone,\
    ElectricityBenchmark, CEREn, HullMarketDataset


def get_model_classes(model_str):
    if model_str == 'spin':
        model, filler = SPINModel, SPINImputer
    elif model_str == 'grin':
        model, filler = GRINModel, Imputer
    elif model_str == 'saits':
        model, filler = SAITS, SAITSImputer
    elif model_str == 'transformer':
        model, filler = TransformerModel, SPINImputer
    elif model_str == 'brits':
        model, filler = BRITS, BRITSImputer
    elif model_str == 'imputeformer':
        model, filler = ImputeFormerModel, ImputeFormerImputer
    else:
        raise ValueError(f'Model {model_str} not available.')
    return model, filler


def get_dataset(dataset_name: str):
    if dataset_name.startswith('air'):
        return AirQuality(impute_nans=True, small=dataset_name[3:] == '36')
    if dataset_name == 'hull':
        return HullMarketDataset()
    # build missing dataset
    if dataset_name.endswith('_point'):
        p_fault, p_noise = 0., 0.25
        dataset_name = dataset_name[:-6]
    elif dataset_name.endswith('_block'):
        p_fault, p_noise = 0.0015, 0.05
        dataset_name = dataset_name[:-6]
    elif dataset_name.endswith('_sparse'):
        p_fault, p_noise = 0., 0.9  # 0.6 0.7, 0.8, 0.9
        dataset_name = dataset_name[:-7]
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}.")
    if dataset_name == 'la':
        return add_missing_values(MetrLA(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=9101112)
    if dataset_name == 'bay':
        return add_missing_values(PemsBay(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    if dataset_name == 'pems03':
        return add_missing_values(PeMS03(mask_zeros=True), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    if dataset_name == 'pems04':
        return add_missing_values(PeMS04(mask_zeros=True), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    if dataset_name == 'pems07':
        return add_missing_values(PeMS07(mask_zeros=True), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    if dataset_name == 'pems08':
        return add_missing_values(PeMS08(mask_zeros=True), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    if dataset_name == 'elergone':
        return add_missing_values(Elergone(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    if dataset_name == 'solar':
        return add_missing_values(SolarBenchmark(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    if dataset_name == 'ecl':
        return add_missing_values(ElectricityBenchmark(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    if dataset_name == 'cer':
        return add_missing_values(CEREn(), p_fault=p_fault, p_noise=p_noise,
                                  min_seq=12, max_seq=12 * 4, seed=56789)
    else:
        raise ValueError(f"Invalid dataset name: {dataset_name}.")


def get_scheduler(scheduler_name: str = None, args=None):
    if scheduler_name is None:
        return None, None
    scheduler_name = scheduler_name.lower()
    if scheduler_name == 'cosine':
        scheduler_class = CosineAnnealingLR
        scheduler_kwargs = dict(eta_min=0.1 * args.lr, T_max=args.epochs)
    elif scheduler_name == 'magic':
        scheduler_class = CosineSchedulerWithRestarts
        scheduler_kwargs = dict(num_warmup_steps=12, min_factor=0.1,
                                linear_decay=0.67,
                                num_training_steps=args.epochs,
                                num_cycles=args.epochs // 100)
    else:
        raise ValueError(f"Invalid scheduler name: {scheduler_name}.")
    return scheduler_class, scheduler_kwargs


def stitch_predictions(torch_dataset, sample_indices, predictions):
    if predictions.shape[0] != len(sample_indices):
        raise ValueError("Prediction count and sample indices do not align.")
    n_steps = torch_dataset.n_steps
    n_nodes = torch_dataset.n_nodes
    n_channels = torch_dataset.n_channels
    agg = np.zeros((n_steps, n_nodes, n_channels), dtype=np.float32)
    counts = np.zeros_like(agg)
    for row_idx, sample_idx in enumerate(sample_indices):
        horizon_idx = torch_dataset.get_horizon_indices(int(sample_idx))
        horizon_idx = horizon_idx.cpu().numpy() if hasattr(horizon_idx, 'cpu') else np.asarray(horizon_idx)
        agg[horizon_idx] += predictions[row_idx]
        counts[horizon_idx] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        averaged = np.divide(agg, counts, out=np.zeros_like(agg), where=counts > 0)
    base = torch_dataset.data.detach().cpu().numpy()
    filled = np.where(counts > 0, averaged, base)
    return filled


def export_hull_imputations(dataset, filled_array, export_path):
    if not hasattr(dataset, 'feature_columns') or not hasattr(dataset, 'date_ids'):
        raise RuntimeError('Export is only supported for HullMarketDataset inputs.')
    feature_cols = list(dataset.feature_columns)
    reshaped = filled_array.reshape(filled_array.shape[0], -1)
    imputed_df = pd.DataFrame(reshaped, columns=feature_cols)
    imputed_df.insert(0, 'date_id', dataset.date_ids)

    raw = pd.read_csv(dataset.csv_path).sort_values('date_id').reset_index(drop=True)
    merged = raw.sort_values('date_id').reset_index(drop=True)
    for col in feature_cols:
        merged[col] = imputed_df[col].values

    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(export_path, index=False)
    logger.info(f'Imputed feature table saved to {export_path}')


def parse_args():
    # Argument parser
    parser = ArgParser()

    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--precision', type=int, default=32)
    parser.add_argument("--model-name", type=str, default='spin')
    parser.add_argument("--dataset-name", type=str, default='air36')
    parser.add_argument("--config", type=str, default='imputation/spin.yaml')
    parser.add_argument('--hull-csv', type=str, default=None,
                        help='Optional override for Hull dataset CSV path.')
    parser.add_argument('--export-imputations', type=str, default=None,
                        help='If provided, save stitched imputations to this CSV path.')

    # Splitting/aggregation params
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)

    # Training params
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--batches-epoch', type=int, default=300)
    parser.add_argument('--batch-inference', type=int, default=32)
    parser.add_argument('--split-batch-in', type=int, default=1)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--loss-fn', type=str, default='l1_loss')
    parser.add_argument('--lr-scheduler', type=str, default=None)

    # Connectivity params
    parser.add_argument("--adj-threshold", type=float, default=0.1)

    known_args, _ = parser.parse_known_args()
    model_cls, imputer_cls = get_model_classes(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = imputer_cls.add_argparse_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = ImputationDataset.add_argparse_args(parser)

    args = parser.parse_args()
    if args.config is not None:
        cfg_path = os.path.join(config.config_dir, args.config)
        with open(cfg_path, 'r') as fp:
            config_args = yaml.load(fp, Loader=yaml.FullLoader)
        for arg in config_args:
            setattr(args, arg, config_args[arg])

    if args.hull_csv:
        os.environ["HULL_MARKET_CSV"] = str(Path(args.hull_csv).expanduser().resolve())

    return args


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(1)
    pl.seed_everything(args.seed)

    # script flags
    model_cls, imputer_class = get_model_classes(args.model_name)
    dataset = get_dataset(args.dataset_name)

    logger.info(args)

    ########################################
    # create logdir and save configuration #
    ########################################

    exp_name = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    exp_name = f"{exp_name}_{args.seed}"
    logdir = os.path.join(config.log_dir, args.dataset_name,
                          args.model_name, exp_name)
    # save config for logging
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, 'config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp,
                  indent=4, sort_keys=True)

    ########################################
    # data module                          #
    ########################################

    # time embedding
    is_spin = args.model_name in ['spin', 'spin_h']
    time_emb = None
    if is_spin or args.model_name == 'transformer':
        time_emb = dataset.datetime_encoded(['day', 'week']).values
        exog_map = {'global_temporal_encoding': time_emb}

        input_map = {
            'u': 'temporal_encoding',
            'x': 'data'
        }
    elif args.model_name == 'imputeformer':
        time_emb = dataset.datetime_encoded(['day']).values
        exog_map = {'global_temporal_encoding': time_emb}

        input_map = {
            'u': 'temporal_encoding',
            'x': 'data'
        }
    else:
        exog_map = input_map = None

    if is_spin or args.model_name == 'grin':
        adj = dataset.get_connectivity(threshold=args.adj_threshold,
                                       include_self=False,
                                       force_symmetric=is_spin)
    elif args.model_name == 'mpgru' or args.model_name == 'bimpgru':
        # get adjacency matrix
        adj = dataset.get_similarity(thr=args.adj_threshold)
        # force adj with no self loop
        np.fill_diagonal(adj, 0.)
    else:
        adj = None

    # instantiate dataset
    torch_dataset = ImputationDataset(*dataset.numpy(return_idx=True),
                                      training_mask=dataset.training_mask,
                                      eval_mask=dataset.eval_mask,
                                      connectivity=adj,
                                      exogenous=exog_map,
                                      input_map=input_map,
                                      window=args.window,
                                      stride=args.stride)

    # get train/val/test indices
    splitter = dataset.get_splitter(val_len=args.val_len,
                                    test_len=args.test_len)

    scalers = {'data': StandardScaler(axis=(0, 1))}

    dm = SpatioTemporalDataModule(torch_dataset,
                                  scalers=scalers,
                                  splitter=splitter,
                                  batch_size=args.batch_size // args.split_batch_in)
    dm.setup()

    ########################################
    # predictor                            #
    ########################################

    time_emb_dim = 0
    if time_emb is not None:
        if time_emb.ndim > 1:
            time_emb_dim = time_emb.shape[1]
        else:
            time_emb_dim = 1

    additional_model_hparams = dict(n_nodes=dm.n_nodes,
                                    input_size=dm.n_channels,
                                    u_size=time_emb_dim,
                                    output_size=dm.n_channels,
                                    windows=dm.window)

    # model's inputs
    model_kwargs = parser_utils.filter_args(
        args={**vars(args), **additional_model_hparams},
        target_cls=model_cls,
        return_dict=True)

    # loss and metrics
    loss_fn = MaskedMetric(metric_fn=getattr(torch.nn.functional, args.loss_fn),
                           compute_on_step=True,
                           metric_kwargs={'reduction': 'none'})

    metrics = {'mae': MaskedMAE(compute_on_step=False),
               'mse': MaskedMSE(compute_on_step=False),
               'mre': MaskedMRE(compute_on_step=False)}

    scheduler_class, scheduler_kwargs = get_scheduler(args.lr_scheduler, args)

    # setup imputer
    imputer_kwargs = parser_utils.filter_argparse_args(args, imputer_class,
                                                       return_dict=True)
    imputer = imputer_class(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': args.lr,
                      'weight_decay': args.l2_reg},
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        **imputer_kwargs
    )

    ########################################
    # training                             #
    ########################################

    # callbacks
    early_stop_callback = EarlyStopping(monitor='val_mae',
                                        patience=args.patience, mode='min')
    checkpoint_callback = ModelCheckpoint(dirpath=logdir, save_top_k=1,
                                          monitor='val_mae', mode='min')

    tb_logger = TensorBoardLogger(logdir, name="model")

    if torch.cuda.is_available():
        accelerator = 'gpu'
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        accelerator = 'mps'
    else:
        accelerator = 'cpu'
    trainer = pl.Trainer(max_epochs=args.epochs,
                         default_root_dir=logdir,
                         logger=tb_logger,
                         precision=args.precision,
                         accumulate_grad_batches=args.split_batch_in,
                         accelerator=accelerator,
                         devices=1,
                         gradient_clip_val=args.grad_clip_val,
                         limit_train_batches=args.batches_epoch * args.split_batch_in,
                         callbacks=[early_stop_callback, checkpoint_callback])

    trainer.fit(imputer,
                train_dataloaders=dm.train_dataloader(),
                val_dataloaders=dm.val_dataloader(
                    batch_size=args.batch_inference))

    ########################################
    # testing                              #
    ########################################

    imputer.load_model(checkpoint_callback.best_model_path)
    imputer.freeze()
    trainer.test(imputer, dataloaders=dm.test_dataloader(
        batch_size=args.batch_inference))

    output = trainer.predict(imputer, dataloaders=dm.test_dataloader(
        batch_size=args.batch_inference))
    output = casting.numpy(output)
    y_hat, y_true, mask = output['y_hat'].squeeze(-1), \
                          output['y'].squeeze(-1), \
                          output['mask'].squeeze(-1)
    check_mae = numpy_metrics.masked_mae(y_hat, y_true, mask)
    print(f'Test MAE: {check_mae:.2f}')
    if args.export_imputations:
        full_loader = StaticGraphLoader(dm.torch_dataset,
                                        batch_size=args.batch_inference,
                                        shuffle=False,
                                        num_workers=dm.workers)
        full_output = trainer.predict(imputer, dataloaders=full_loader)
        full_output = casting.numpy(full_output)
        full_preds = full_output['y_hat'].squeeze(-1)
        sample_indices = np.arange(len(dm.torch_dataset))
        stitched = stitch_predictions(dm.torch_dataset, sample_indices, full_preds)
        export_hull_imputations(dataset, stitched, args.export_imputations)
    return y_hat


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)
