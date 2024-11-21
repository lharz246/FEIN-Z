class HParamsView(object):
    def __init__(self, d):
        self.__dict__ = d


def create_hparams(**kwargs):
    """Create spk_embedder hyperparameters. Parse nondefault from given string."""

    hparams = {
        ################################
        # Experiment Parameters        #
        ################################
        "iters_per_checkpoint": 15000,
        "seed": 16807,
        "fp16_run": False,
        "cudnn_enabled": True,
        "cudnn_benchmark": True,
        "output_directory": "dyadic_bcn2",  # Directory to save checkpoints.
        "log_directory": 'log',
        "checkpoint_path": '/media/compute/homes/lharz/genea23/genea_challenge_2023-main/baselines/2023_ivi_baseline-main/Tacotron2/dyadic_bcn2/ckpt/checkpoint_70000.pt',
        "warm_start": True,
        "device": 0,
        "n_acoustic_feat_dims": 168,
        "warmup_period": 50,
        "dyadic": True,
        "sequence_length": 250,
        ################################
        # Model Parameters             #
        ################################

        ################################
        # Audio Encoder                #
        ################################
        "model_monadic": HParamsView({
            "tae": HParamsView({
                "text_audio_attention": HParamsView({
                    "conv_in": 427,
                    "kernel_size": 1,
                    "embed_dim": 512,
                    "num_heads": 8,
                    "dropout": 0.3,
                    "bias": False,
                    "feedforward": HParamsView({
                        "in_dim": 512,
                        "lin_dim": 512,
                        "out_dim": 512,
                    })
                }),
                "conv_in": 512,
                "conv_dim1": 256,
                "conv_dim": 512,
                "kernel_size": 1,

            }),
            "beta_attention": HParamsView({
                "in_dim": 512,
                "lin_dim": 256,
                "out_dim": 128,
            }),
            "gamma_attention": HParamsView({
                "in_dim": 512,
                "lin_dim": 256,
                "out_dim": 256,
            }),
            "gesture_attention": HParamsView({
                "conv_in": 168,
                "kernel_size": 1,
                "embed_dim": 512,
                "num_heads": 8,
                "dropout": 0.3,
                "bias": False,
                "feedforward": HParamsView({
                    "in_dim": 512,
                    "lin_dim": 256,
                    "out_dim": 512,
                })
            }),

            "fein": HParamsView({
                "gesture_attention": HParamsView({
                    "conv_in": 512,
                    "kernel_size": 1,
                    "embed_dim": 512,
                    "num_heads": 8,
                    "dropout": 0.3,
                    "bias": False,
                    "feedforward": HParamsView({
                        "in_dim": 512,
                        "lin_dim": 1024,
                        "out_dim": 512,
                    })
                }),
                "conv_in": 50,
                "conv_dim1": 200,
                "conv_dim2": 168,
                "conv_dim3": 256,
                "conv_dim4": 512,
                "kernel_size_in": 3,
                "kernel_size": 1,
                "in_dim": 512,
                "in_dim_mem": 1024,
                "lin_dim": 512,
                "gamnma_out": 512,
                "memory": 512,
                "beta_out": 512
            }),

            "control_network": HParamsView({
                "control_conv": 512,
                "lin_in": 256,
                "lin_dim": 128,
                "lin_out": 128,
                "control_dim": 256,
                "controllable_joints": [24, 12, 66, 66]
                # "controllable_joints": [24, 12, 66, 66]
            })

        }),
        "model_dyadic": HParamsView({
            "tae": HParamsView({
                "text_audio_attention": HParamsView({
                    "conv_in": 854,
                    "kernel_size": 1,
                    "embed_dim": 512,
                    "num_heads": 8,
                    "dropout": 0.3,
                    "bias": False,
                    "feedforward": HParamsView({
                        "in_dim": 512,
                        "lin_dim": 512,
                        "out_dim": 512,
                    })
                }),
                "conv_in": 512,
                "conv_dim1": 256,
                "conv_dim": 512,
                "kernel_size": 1,

            }),
            "beta_attention": HParamsView({
                "in_dim": 512,
                "lin_dim": 256,
                "out_dim": 128,
            }),
            "gamma_attention": HParamsView({
                "in_dim": 512,
                "lin_dim": 256,
                "out_dim": 256,
            }),
            "gesture_attention": HParamsView({
                "conv_in": 336,
                "kernel_size": 1,
                "embed_dim": 512,
                "num_heads": 8,
                "dropout": 0.3,
                "bias": False,
                "feedforward": HParamsView({
                    "in_dim": 512,
                    "lin_dim": 256,
                    "out_dim": 512,
                })
            }),

            "fein": HParamsView({
                "gesture_attention": HParamsView({
                    "conv_in": 512,
                    "kernel_size": 1,
                    "embed_dim": 512,
                    "num_heads": 8,
                    "dropout": 0.3,
                    "bias": False,
                    "feedforward": HParamsView({
                        "in_dim": 512,
                        "lin_dim": 1024,
                        "out_dim": 512,
                    })
                }),
                "conv_in": 50,
                "conv_dim1": 200,
                "conv_dim2": 336,
                "conv_dim3": 256,
                "conv_dim4": 512,
                "kernel_size_in": 3,
                "kernel_size": 1,
                "in_dim": 512,
                "in_dim_mem": 1024,
                "lin_dim": 512,
                "gamnma_out": 512,
                "memory": 512,
                "beta_out": 512
            }),

            "control_network": HParamsView({
                "control_conv": 512,
                "lin_in": 256,
                "lin_dim": 128,
                "lin_out": 128,
                "control_dim": 256,
                "controllable_joints": [24, 12, 66, 66, 24, 12, 66, 66]
                # "controllable_joints": [24, 12, 66, 66]
            })
        }),
        "discriminator": HParamsView({
            "action_dim": 168,
        }),

        ################################
        # Optimization Hyperparameters #
        ################################
        "use_saved_learning_rate": False,
        "learning_rate": 5e-5,
        "leaning_rate_disc": 1e-4,
        "weight_decay": 0.01,
        "grad_clip_thresh": 1.0,
        "batch_size": 128,
        "mask_padding": True,  # set spk_embedder's padded outputs to padded values
        "mel_weight": 1,
        "gate_weight": 0,
        "vel_weight": 1,
        "pos_weight": 0.01,
        "add_l1_losss": False
    }

    for key, val in kwargs.items():
        if key in hparams:
            hparams[key] = val
        else:
            raise ValueError('The hyper-parameter %s is not supported.' % key)

    hparams_view = HParamsView(hparams)

    return hparams_view
