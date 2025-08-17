[I 2025-07-23 01:13:37,308] Trial 3 finished with value: 5.644055366516113 and parameters: {'n_steps': 128, 'gamma': 0.85, 'learning_rate': 0.0006634631819583468, 'target_kl': 0.005, 'gae_lambda': 0.98, 'batch_size': 512, 'ent_coef': 0.0004}. Best is trial 2 with value: 6.5730509757995605.
[I 2025-07-23 02:42:22,673] Trial 4 finished with value: 5.102229118347168 and parameters: {'n_steps': 512, 'gamma': 0.9, 'learning_rate': 0.034435280556949686, 'target_kl': 0.03, 'gae_lambda': 0.99, 'batch_size': 8, 'ent_coef': 0.0007}. Best is trial 2 with value: 6.5730509757995605.
[I 2025-07-23 03:17:11,946] Trial 5 finished with value: 5.174429893493652 and parameters: {'n_steps': 1024, 'gamma': 0.8, 'learning_rate': 0.00013716597556496702, 'target_kl': 0.1, 'gae_lambda': 0.9, 'batch_size': 512, 'ent_coef': 0.001}. Best is trial 2 with value: 6.5730509757995605.
[I 2025-07-23 04:02:26,076] Trial 6 finished with value: 6.213438034057617 and parameters: {'n_steps': 16, 'gamma': 0.9, 'learning_rate': 0.2658617021690577, 'target_kl': 0.01, 'gae_lambda': 0.8, 'batch_size': 32, 'ent_coef': 0.0009000000000000001}. Best is trial 2 with value: 6.5730509757995605.
/Users/dach/dev/deep-rl/Omega.py:78: UserWarning: Creating legend with loc="best" can be slow with large amounts of data.
  plt.tight_layout()
[I 2025-07-23 04:36:04,707] Trial 7 finished with value: 4.988029479980469 and parameters: {'n_steps': 256, 'gamma': 0.8, 'learning_rate': 0.025982873430872334, 'target_kl': 0.1, 'gae_lambda': 1.0, 'batch_size': 16, 'ent_coef': 0.0}. Best is trial 2 with value: 6.5730509757995605.
/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/sb3_contrib/trpo/trpo.py:154: UserWarning: You have specified a mini-batch size of 128, but because the `RolloutBuffer` is of size `n_steps * n_envs = 32`, after every 0 untruncated mini-batches, there will be a truncated mini-batch of size 32
We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.
Info: (n_steps=32 and n_envs=1)
  warnings.warn(
[I 2025-07-23 05:40:58,960] Trial 8 finished with value: 5.680668830871582 and parameters: {'n_steps': 32, 'gamma': 0.95, 'learning_rate': 0.0001940223909473507, 'target_kl': 0.03, 'gae_lambda': 0.99, 'batch_size': 64, 'ent_coef': 0.0009000000000000001}. Best is trial 2 with value: 6.5730509757995605.
[I 2025-07-23 06:30:18,856] Trial 9 finished with value: 6.576828956604004 and parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 0.005013644107864597, 'target_kl': 0.02, 'gae_lambda': 0.92, 'batch_size': 512, 'ent_coef': 0.0004}. Best is trial 9 with value: 6.576828956604004.
Completed 10 trials for trpor_no_noise.

Running batch 1 for trpor_with_noise (10 trials for 1000000 timesteps)...
/Users/dach/dev/deep-rl/Omega.py:79: UserWarning: Creating legend with loc="best" can be slow with large amounts of data.
  plt.savefig(os.path.join(self.log_dir, "graph.png"))
[I 2025-07-23 07:29:17,841] Trial 0 finished with value: 5.235568046569824 and parameters: {'n_steps': 2048, 'gamma': 0.95, 'learning_rate': 0.11900110702790527, 'target_kl': 0.05, 'gae_lambda': 0.99, 'batch_size': 1024, 'ent_coef': 0.0001}. Best is trial 0 with value: 5.235568046569824.
[I 2025-07-23 11:56:24,422] Trial 1 finished with value: 5.022815704345703 and parameters: {'n_steps': 2048, 'gamma': 0.9, 'learning_rate': 0.014881955161923716, 'target_kl': 0.03, 'gae_lambda': 0.92, 'batch_size': 8, 'ent_coef': 0.0002}. Best is trial 0 with value: 5.235568046569824.
[I 2025-07-23 14:58:20,915] Trial 2 finished with value: 5.449995517730713 and parameters: {'n_steps': 128, 'gamma': 0.99, 'learning_rate': 0.005574209822503177, 'target_kl': 0.02, 'gae_lambda': 0.98, 'batch_size': 512, 'ent_coef': 0.00030000000000000003}. Best is trial 2 with value: 5.449995517730713.
[I 2025-07-23 15:45:59,539] Trial 3 finished with value: 5.6962056159973145 and parameters: {'n_steps': 16, 'gamma': 0.9, 'learning_rate': 0.1300173269714677, 'target_kl': 0.005, 'gae_lambda': 0.98, 'batch_size': 512, 'ent_coef': 0.0007}. Best is trial 3 with value: 5.6962056159973145.
[I 2025-07-23 17:07:09,287] Trial 4 finished with value: 5.362671852111816 and parameters: {'n_steps': 128, 'gamma': 0.85, 'learning_rate': 0.0039374630218307624, 'target_kl': 0.001, 'gae_lambda': 0.98, 'batch_size': 16, 'ent_coef': 0.0001}. Best is trial 3 with value: 5.6962056159973145.
[I 2025-07-23 17:58:59,554] Trial 5 finished with value: 6.343036651611328 and parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 0.0005089479641399226, 'target_kl': 0.03, 'gae_lambda': 0.9, 'batch_size': 256, 'ent_coef': 0.0001}. Best is trial 5 with value: 6.343036651611328.
[I 2025-07-23 18:47:06,230] Trial 6 finished with value: 5.614182472229004 and parameters: {'n_steps': 128, 'gamma': 0.9, 'learning_rate': 4.922794808829914e-05, 'target_kl': 0.05, 'gae_lambda': 0.99, 'batch_size': 64, 'ent_coef': 0.0006000000000000001}. Best is trial 5 with value: 6.343036651611328.
[I 2025-07-23 20:16:52,380] Trial 7 finished with value: 5.3261919021606445 and parameters: {'n_steps': 512, 'gamma': 0.9, 'learning_rate': 1.0675990387785885e-05, 'target_kl': 0.02, 'gae_lambda': 0.8, 'batch_size': 32, 'ent_coef': 0.0004}. Best is trial 5 with value: 6.343036651611328.
[I 2025-07-23 21:24:58,257] Trial 8 finished with value: 5.083028793334961 and parameters: {'n_steps': 1024, 'gamma': 0.99, 'learning_rate': 0.8447353239736433, 'target_kl': 0.1, 'gae_lambda': 0.9, 'batch_size': 256, 'ent_coef': 0.0001}. Best is trial 5 with value: 6.343036651611328.
[I 2025-07-23 22:34:04,619] Trial 9 finished with value: 5.622033596038818 and parameters: {'n_steps': 128, 'gamma': 0.95, 'learning_rate': 0.00014366061648914263, 'target_kl': 0.005, 'gae_lambda': 0.99, 'batch_size': 512, 'ent_coef': 0.0001}. Best is trial 5 with value: 6.343036651611328.
Completed 10 trials for trpor_with_noise.

Best Trial Stats Across All Configs (Batch 2):
TRPO NO NOISE:
  Best Parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 1.9809257913083674e-05, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 128}
  Best Max Reward: 6.84
TRPO WITH NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.02889732147809502, 'target_kl': 0.05, 'gae_lambda': 0.99, 'batch_size': 16}
  Best Max Reward: 6.47
TRPOR NO NOISE:
  Best Parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 0.005013644107864597, 'target_kl': 0.02, 'gae_lambda': 0.92, 'batch_size': 512, 'ent_coef': 0.0004}
  Best Max Reward: 6.58
TRPOR WITH NOISE:
  Best Parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 0.0005089479641399226, 'target_kl': 0.03, 'gae_lambda': 0.9, 'batch_size': 256, 'ent_coef': 0.0001}
  Best Max Reward: 6.34

Running batch 2 for trpo_no_noise (10 trials for 1000000 timesteps)...
[I 2025-07-24 00:07:52,408] Trial 10 finished with value: 6.6633524894714355 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 3.501417316098312e-05, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 8 with value: 6.8364458084106445.
[I 2025-07-24 01:32:54,510] Trial 11 finished with value: 6.669475078582764 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 1.0795611739060677e-05, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 8 with value: 6.8364458084106445.
[I 2025-07-24 03:06:46,208] Trial 12 finished with value: 6.818927764892578 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 1.0310224600885922e-05, 'target_kl': 0.001, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 8 with value: 6.8364458084106445.
[I 2025-07-24 04:35:39,164] Trial 13 finished with value: 5.2646026611328125 and parameters: {'n_steps': 512, 'gamma': 0.95, 'learning_rate': 2.0239764646389996e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 8}. Best is trial 8 with value: 6.8364458084106445.
[I 2025-07-24 04:46:40,036] Trial 14 finished with value: 5.087845325469971 and parameters: {'n_steps': 1024, 'gamma': 0.8, 'learning_rate': 0.03576860690625641, 'target_kl': 0.001, 'gae_lambda': 0.8, 'batch_size': 512}. Best is trial 8 with value: 6.8364458084106445.
[I 2025-07-24 06:23:22,531] Trial 15 finished with value: 6.456174373626709 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.0002710329768610387, 'target_kl': 0.001, 'gae_lambda': 0.8, 'batch_size': 128}. Best is trial 8 with value: 6.8364458084106445.
[I 2025-07-24 07:30:52,366] Trial 16 finished with value: 6.491371154785156 and parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 4.899539747365285e-05, 'target_kl': 0.01, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 8 with value: 6.8364458084106445.
[I 2025-07-24 08:54:18,506] Trial 17 finished with value: 5.54202938079834 and parameters: {'n_steps': 2048, 'gamma': 0.95, 'learning_rate': 1.0004307406157486e-05, 'target_kl': 0.02, 'gae_lambda': 0.98, 'batch_size': 8}. Best is trial 8 with value: 6.8364458084106445.
[I 2025-07-24 10:31:42,881] Trial 18 finished with value: 5.233576774597168 and parameters: {'n_steps': 1024, 'gamma': 0.95, 'learning_rate': 0.001600842647371623, 'target_kl': 0.05, 'gae_lambda': 1.0, 'batch_size': 512}. Best is trial 8 with value: 6.8364458084106445.
[I 2025-07-24 11:10:45,251] Trial 19 finished with value: 5.475851058959961 and parameters: {'n_steps': 128, 'gamma': 0.85, 'learning_rate': 0.00015779486079463018, 'target_kl': 0.03, 'gae_lambda': 0.99, 'batch_size': 64}. Best is trial 8 with value: 6.8364458084106445.
Completed 10 trials for trpo_no_noise.

Running batch 2 for trpo_with_noise (10 trials for 1000000 timesteps)...
[I 2025-07-24 11:30:29,146] Trial 10 finished with value: 4.878791809082031 and parameters: {'n_steps': 512, 'gamma': 0.8, 'learning_rate': 0.769224196792309, 'target_kl': 0.05, 'gae_lambda': 0.99, 'batch_size': 16}. Best is trial 2 with value: 6.467962741851807.
[I 2025-07-24 12:48:32,718] Trial 11 finished with value: 6.494507789611816 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.15645677429925703, 'target_kl': 0.03, 'gae_lambda': 0.99, 'batch_size': 16}. Best is trial 11 with value: 6.494507789611816.
[I 2025-07-24 14:05:39,330] Trial 12 finished with value: 6.494594573974609 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.04931716877455356, 'target_kl': 0.03, 'gae_lambda': 0.99, 'batch_size': 16}. Best is trial 12 with value: 6.494594573974609.
[I 2025-07-24 15:25:18,645] Trial 13 finished with value: 6.284100532531738 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.08928661921052067, 'target_kl': 0.03, 'gae_lambda': 0.99, 'batch_size': 8}. Best is trial 12 with value: 6.494594573974609.
[I 2025-07-24 15:57:25,790] Trial 14 finished with value: 5.326526641845703 and parameters: {'n_steps': 128, 'gamma': 0.9, 'learning_rate': 0.6613466630266871, 'target_kl': 0.03, 'gae_lambda': 0.99, 'batch_size': 16}. Best is trial 12 with value: 6.494594573974609.
[I 2025-07-24 17:17:13,973] Trial 15 finished with value: 6.436241149902344 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 0.017987027205974542, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 16}. Best is trial 12 with value: 6.494594573974609.
[I 2025-07-24 18:37:17,676] Trial 16 finished with value: 6.477451801300049 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.1399748073552988, 'target_kl': 0.03, 'gae_lambda': 0.98, 'batch_size': 16}. Best is trial 12 with value: 6.494594573974609.
[I 2025-07-24 19:00:34,467] Trial 17 finished with value: 5.351114273071289 and parameters: {'n_steps': 256, 'gamma': 0.85, 'learning_rate': 0.00753323095931199, 'target_kl': 0.03, 'gae_lambda': 1.0, 'batch_size': 128}. Best is trial 12 with value: 6.494594573974609.
[I 2025-07-24 19:28:16,752] Trial 18 finished with value: 5.661746501922607 and parameters: {'n_steps': 64, 'gamma': 0.85, 'learning_rate': 0.09615819608797688, 'target_kl': 0.02, 'gae_lambda': 0.92, 'batch_size': 1024}. Best is trial 12 with value: 6.494594573974609.
[I 2025-07-24 20:00:41,251] Trial 19 finished with value: 4.940674304962158 and parameters: {'n_steps': 512, 'gamma': 0.9, 'learning_rate': 0.21089504914414742, 'target_kl': 0.1, 'gae_lambda': 0.9, 'batch_size': 8}. Best is trial 12 with value: 6.494594573974609.
Completed 10 trials for trpo_with_noise.

Running batch 2 for trpor_no_noise (10 trials for 1000000 timesteps)...
/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/sb3_contrib/trpo/trpo.py:154: UserWarning: You have specified a mini-batch size of 128, but because the `RolloutBuffer` is of size `n_steps * n_envs = 64`, after every 0 untruncated mini-batches, there will be a truncated mini-batch of size 64
We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.
Info: (n_steps=64 and n_envs=1)
  warnings.warn(
[I 2025-07-24 21:01:29,085] Trial 10 finished with value: 5.839235782623291 and parameters: {'n_steps': 64, 'gamma': 0.99, 'learning_rate': 0.0025597988277143466, 'target_kl': 0.02, 'gae_lambda': 0.92, 'batch_size': 128, 'ent_coef': 0.0005}. Best is trial 9 with value: 6.576828956604004.
[I 2025-07-24 22:12:19,839] Trial 11 finished with value: 6.433170318603516 and parameters: {'n_steps': 16, 'gamma': 0.85, 'learning_rate': 1.2934716591376132e-05, 'target_kl': 0.05, 'gae_lambda': 0.92, 'batch_size': 1024, 'ent_coef': 0.0007}. Best is trial 9 with value: 6.576828956604004.
[I 2025-07-24 23:31:51,766] Trial 12 finished with value: 7.1592116355896 and parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 2.0623809231485985e-05, 'target_kl': 0.001, 'gae_lambda': 0.95, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 12 with value: 7.1592116355896.
[I 2025-07-25 01:17:25,942] Trial 13 finished with value: 5.0358567237854 and parameters: {'n_steps': 2048, 'gamma': 0.95, 'learning_rate': 0.003862496650008723, 'target_kl': 0.001, 'gae_lambda': 0.95, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 12 with value: 7.1592116355896.
[I 2025-07-25 02:30:35,203] Trial 14 finished with value: 6.475635528564453 and parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 0.004839278217798359, 'target_kl': 0.02, 'gae_lambda': 0.95, 'batch_size': 512, 'ent_coef': 0.0001}. Best is trial 12 with value: 7.1592116355896.
[I 2025-07-25 03:13:32,030] Trial 15 finished with value: 6.277042865753174 and parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 0.9672439532849503, 'target_kl': 0.001, 'gae_lambda': 0.95, 'batch_size': 32, 'ent_coef': 0.0005}. Best is trial 12 with value: 7.1592116355896.
[I 2025-07-25 06:32:22,822] Trial 16 finished with value: 5.133453369140625 and parameters: {'n_steps': 2048, 'gamma': 0.99, 'learning_rate': 6.4936418938624e-05, 'target_kl': 0.02, 'gae_lambda': 0.98, 'batch_size': 128, 'ent_coef': 0.00030000000000000003}. Best is trial 12 with value: 7.1592116355896.
[I 2025-07-25 08:28:12,586] Trial 17 finished with value: 6.140834331512451 and parameters: {'n_steps': 32, 'gamma': 0.95, 'learning_rate': 0.0010955832731162046, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0006000000000000001}. Best is trial 12 with value: 7.1592116355896.
[I 2025-07-25 10:03:13,667] Trial 18 finished with value: 6.935138702392578 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 5.4221639937275095e-05, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 8, 'ent_coef': 0.00030000000000000003}. Best is trial 12 with value: 7.1592116355896.
[I 2025-07-25 12:04:15,537] Trial 19 finished with value: 6.756748199462891 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 4.20830642165594e-05, 'target_kl': 0.005, 'gae_lambda': 0.95, 'batch_size': 8, 'ent_coef': 0.0001}. Best is trial 12 with value: 7.1592116355896.
Completed 10 trials for trpor_no_noise.

Running batch 2 for trpor_with_noise (10 trials for 1000000 timesteps)...
[I 2025-07-25 12:48:09,961] Trial 10 finished with value: 5.345527648925781 and parameters: {'n_steps': 256, 'gamma': 0.8, 'learning_rate': 0.0004597259314284153, 'target_kl': 0.03, 'gae_lambda': 0.9, 'batch_size': 256, 'ent_coef': 0.001}. Best is trial 5 with value: 6.343036651611328.
[I 2025-07-25 13:28:45,460] Trial 11 finished with value: 6.056785583496094 and parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 0.034603082988485585, 'target_kl': 0.01, 'gae_lambda': 0.95, 'batch_size': 512, 'ent_coef': 0.0007}. Best is trial 5 with value: 6.343036651611328.
[I 2025-07-25 15:11:19,715] Trial 12 finished with value: 6.3394670486450195 and parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 0.0009176335538561488, 'target_kl': 0.01, 'gae_lambda': 0.95, 'batch_size': 128, 'ent_coef': 0.0008}. Best is trial 5 with value: 6.343036651611328.
[I 2025-07-25 16:35:25,475] Trial 13 finished with value: 6.249434471130371 and parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 0.0008734539058927128, 'target_kl': 0.01, 'gae_lambda': 1.0, 'batch_size': 128, 'ent_coef': 0.001}. Best is trial 5 with value: 6.343036651611328.
[I 2025-07-25 17:37:19,643] Trial 14 finished with value: 6.329526901245117 and parameters: {'n_steps': 32, 'gamma': 0.95, 'learning_rate': 0.0010027989672731956, 'target_kl': 0.01, 'gae_lambda': 0.95, 'batch_size': 128, 'ent_coef': 0.0008}. Best is trial 5 with value: 6.343036651611328.
[I 2025-07-25 18:37:59,126] Trial 15 finished with value: 6.530351638793945 and parameters: {'n_steps': 16, 'gamma': 0.8, 'learning_rate': 0.0001109109247113158, 'target_kl': 0.03, 'gae_lambda': 0.95, 'batch_size': 256, 'ent_coef': 0.0005}. Best is trial 15 with value: 6.530351638793945.
[I 2025-07-25 20:05:05,053] Trial 16 finished with value: 6.791697025299072 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 4.311148904604061e-05, 'target_kl': 0.03, 'gae_lambda': 0.9, 'batch_size': 256, 'ent_coef': 0.0004}. Best is trial 16 with value: 6.791697025299072.
[I 2025-07-25 21:21:39,485] Trial 17 finished with value: 6.397900104522705 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 1.5551917575768134e-05, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 256, 'ent_coef': 0.0005}. Best is trial 16 with value: 6.791697025299072.
[I 2025-07-25 21:44:59,763] Trial 18 finished with value: 5.663063049316406 and parameters: {'n_steps': 64, 'gamma': 0.8, 'learning_rate': 7.223578025490293e-05, 'target_kl': 0.03, 'gae_lambda': 0.92, 'batch_size': 256, 'ent_coef': 0.0004}. Best is trial 16 with value: 6.791697025299072.
[I 2025-07-25 23:03:12,990] Trial 19 finished with value: 7.036706447601318 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 0.00016668682958190787, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.0005}. Best is trial 19 with value: 7.036706447601318.
Completed 10 trials for trpor_with_noise.

Best Trial Stats Across All Configs (Batch 3):
TRPO NO NOISE:
  Best Parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 1.9809257913083674e-05, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 128}
  Best Max Reward: 6.84
TRPO WITH NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.04931716877455356, 'target_kl': 0.03, 'gae_lambda': 0.99, 'batch_size': 16}
  Best Max Reward: 6.49
TRPOR NO NOISE:
  Best Parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 2.0623809231485985e-05, 'target_kl': 0.001, 'gae_lambda': 0.95, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}
  Best Max Reward: 7.16
TRPOR WITH NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 0.00016668682958190787, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.0005}
  Best Max Reward: 7.04

Running batch 3 for trpo_no_noise (10 trials for 1000000 timesteps)...
[I 2025-07-25 23:55:42,574] Trial 20 finished with value: 5.287248611450195 and parameters: {'n_steps': 512, 'gamma': 0.99, 'learning_rate': 2.6367196410568118e-05, 'target_kl': 0.001, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 8 with value: 6.8364458084106445.
[I 2025-07-26 01:37:43,531] Trial 21 finished with value: 7.419168949127197 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 1.0145975538440495e-05, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-26 03:38:33,820] Trial 22 finished with value: 6.728252410888672 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 5.900106856596327e-05, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-26 04:57:12,000] Trial 23 finished with value: 6.460599899291992 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 1.3004844253667098e-05, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-26 07:04:14,603] Trial 24 finished with value: 5.769428253173828 and parameters: {'n_steps': 64, 'gamma': 0.95, 'learning_rate': 4.0313765390602634e-05, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 128}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-26 07:57:17,889] Trial 25 finished with value: 6.443744659423828 and parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 0.00027897770325874835, 'target_kl': 0.001, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-26 09:11:56,007] Trial 26 finished with value: 6.8731536865234375 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 7.894429369918283e-05, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 128}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-26 10:31:03,937] Trial 27 finished with value: 6.624092102050781 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 9.057756942218768e-05, 'target_kl': 0.03, 'gae_lambda': 0.9, 'batch_size': 128}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-26 11:12:34,710] Trial 28 finished with value: 5.869847297668457 and parameters: {'n_steps': 16, 'gamma': 0.8, 'learning_rate': 0.0046083698053987696, 'target_kl': 0.03, 'gae_lambda': 0.95, 'batch_size': 128}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-26 12:04:08,798] Trial 29 finished with value: 6.708075046539307 and parameters: {'n_steps': 16, 'gamma': 0.8, 'learning_rate': 0.00024392541938532582, 'target_kl': 0.01, 'gae_lambda': 0.99, 'batch_size': 128}. Best is trial 21 with value: 7.419168949127197.
Completed 10 trials for trpo_no_noise.

Running batch 3 for trpo_with_noise (10 trials for 1000000 timesteps)...
[I 2025-07-26 12:14:45,225] Trial 20 finished with value: 4.978120803833008 and parameters: {'n_steps': 2048, 'gamma': 0.8, 'learning_rate': 0.04229546757784894, 'target_kl': 0.03, 'gae_lambda': 0.99, 'batch_size': 64}. Best is trial 12 with value: 6.494594573974609.
[I 2025-07-26 13:30:00,152] Trial 21 finished with value: 6.448094844818115 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.46441872572452386, 'target_kl': 0.03, 'gae_lambda': 0.98, 'batch_size': 16}. Best is trial 12 with value: 6.494594573974609.
[I 2025-07-26 14:46:46,708] Trial 22 finished with value: 6.517463684082031 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.24092835044741195, 'target_kl': 0.03, 'gae_lambda': 0.98, 'batch_size': 16}. Best is trial 22 with value: 6.517463684082031.
[I 2025-07-26 16:03:57,656] Trial 23 finished with value: 5.914375305175781 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.323237418807295, 'target_kl': 0.03, 'gae_lambda': 0.98, 'batch_size': 16}. Best is trial 22 with value: 6.517463684082031.
[I 2025-07-26 16:57:01,433] Trial 24 finished with value: 5.096961975097656 and parameters: {'n_steps': 1024, 'gamma': 0.85, 'learning_rate': 0.04932854222893684, 'target_kl': 0.03, 'gae_lambda': 0.99, 'batch_size': 32}. Best is trial 22 with value: 6.517463684082031.
[I 2025-07-26 18:02:01,088] Trial 25 finished with value: 5.348243713378906 and parameters: {'n_steps': 128, 'gamma': 0.99, 'learning_rate': 0.008661765544173434, 'target_kl': 0.03, 'gae_lambda': 0.98, 'batch_size': 16}. Best is trial 22 with value: 6.517463684082031.
[I 2025-07-26 19:18:48,037] Trial 26 finished with value: 6.109005451202393 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.8609686665941041, 'target_kl': 0.03, 'gae_lambda': 0.99, 'batch_size': 16}. Best is trial 22 with value: 6.517463684082031.
[I 2025-07-26 20:47:05,813] Trial 27 finished with value: 6.311675071716309 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.06662803285627883, 'target_kl': 0.02, 'gae_lambda': 0.9, 'batch_size': 16}. Best is trial 22 with value: 6.517463684082031.
[I 2025-07-26 22:06:46,068] Trial 28 finished with value: 6.317296504974365 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.009427341533471955, 'target_kl': 0.05, 'gae_lambda': 0.92, 'batch_size': 16}. Best is trial 22 with value: 6.517463684082031.
[I 2025-07-26 22:42:31,779] Trial 29 finished with value: 5.944093227386475 and parameters: {'n_steps': 32, 'gamma': 0.85, 'learning_rate': 1.4668869529603435e-05, 'target_kl': 0.1, 'gae_lambda': 1.0, 'batch_size': 1024}. Best is trial 22 with value: 6.517463684082031.
Completed 10 trials for trpo_with_noise.

Running batch 3 for trpor_no_noise (10 trials for 1000000 timesteps)...
[I 2025-07-27 00:27:28,372] Trial 20 finished with value: 6.777178764343262 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 3.6255900708289634e-05, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 8, 'ent_coef': 0.0002}. Best is trial 12 with value: 7.1592116355896.
[I 2025-07-27 02:17:26,840] Trial 21 finished with value: 6.645625114440918 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 3.454830901493056e-05, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 8, 'ent_coef': 0.0002}. Best is trial 12 with value: 7.1592116355896.
[I 2025-07-27 05:00:40,287] Trial 22 finished with value: 6.914755821228027 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.00031461649067098565, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 8, 'ent_coef': 0.00030000000000000003}. Best is trial 12 with value: 7.1592116355896.
[I 2025-07-27 06:32:17,699] Trial 23 finished with value: 6.897218227386475 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.0002751859037256171, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 8, 'ent_coef': 0.0004}. Best is trial 12 with value: 7.1592116355896.
[I 2025-07-27 07:11:50,740] Trial 24 finished with value: 5.2043256759643555 and parameters: {'n_steps': 256, 'gamma': 0.8, 'learning_rate': 0.0004953880070449991, 'target_kl': 0.05, 'gae_lambda': 0.95, 'batch_size': 64, 'ent_coef': 0.00030000000000000003}. Best is trial 12 with value: 7.1592116355896.
[I 2025-07-27 08:56:06,434] Trial 25 finished with value: 5.923716068267822 and parameters: {'n_steps': 64, 'gamma': 0.9, 'learning_rate': 8.792733235109744e-05, 'target_kl': 0.01, 'gae_lambda': 0.92, 'batch_size': 1024, 'ent_coef': 0.0}. Best is trial 12 with value: 7.1592116355896.
[I 2025-07-27 13:50:24,288] Trial 26 finished with value: 5.330818176269531 and parameters: {'n_steps': 128, 'gamma': 0.99, 'learning_rate': 2.064528246116658e-05, 'target_kl': 0.001, 'gae_lambda': 0.8, 'batch_size': 8, 'ent_coef': 0.0005}. Best is trial 12 with value: 7.1592116355896.
[I 2025-07-27 15:57:35,536] Trial 27 finished with value: 6.5188117027282715 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.00033670362437314595, 'target_kl': 0.005, 'gae_lambda': 0.99, 'batch_size': 8, 'ent_coef': 0.0001}. Best is trial 12 with value: 7.1592116355896.
[I 2025-07-27 17:43:04,882] Trial 28 finished with value: 6.42375373840332 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.00010894604642781703, 'target_kl': 0.005, 'gae_lambda': 0.9, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 12 with value: 7.1592116355896.
[I 2025-07-27 21:12:02,922] Trial 29 finished with value: 4.956664562225342 and parameters: {'n_steps': 1024, 'gamma': 0.95, 'learning_rate': 0.0011461283618238057, 'target_kl': 0.001, 'gae_lambda': 0.98, 'batch_size': 8, 'ent_coef': 0.0002}. Best is trial 12 with value: 7.1592116355896.
Completed 10 trials for trpor_no_noise.

Running batch 3 for trpor_with_noise (10 trials for 1000000 timesteps)...
[I 2025-07-27 23:07:16,074] Trial 20 finished with value: 7.188906669616699 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 2.3222247001289904e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 20 with value: 7.188906669616699.
[I 2025-07-28 00:40:32,356] Trial 21 finished with value: 7.099003791809082 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 2.859268907320019e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 20 with value: 7.188906669616699.
[I 2025-07-28 02:09:33,810] Trial 22 finished with value: 6.996152877807617 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 2.4582488897676093e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 20 with value: 7.188906669616699.
[I 2025-07-28 03:42:10,386] Trial 23 finished with value: 6.631361961364746 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 0.000215156075564254, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 20 with value: 7.188906669616699.
[I 2025-07-28 05:02:11,915] Trial 24 finished with value: 6.820412635803223 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 2.4947642316661034e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.0}. Best is trial 20 with value: 7.188906669616699.
[I 2025-07-28 06:40:48,408] Trial 25 finished with value: 6.6059489250183105 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 0.0002470543606942871, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.0006000000000000001}. Best is trial 20 with value: 7.188906669616699.
[I 2025-07-28 08:26:35,769] Trial 26 finished with value: 6.744350433349609 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 1.0031821180584155e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.0002}. Best is trial 20 with value: 7.188906669616699.
[I 2025-07-28 09:22:05,251] Trial 27 finished with value: 5.282442569732666 and parameters: {'n_steps': 512, 'gamma': 0.8, 'learning_rate': 4.708053120968668e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.0005}. Best is trial 20 with value: 7.188906669616699.
[I 2025-07-28 10:15:41,856] Trial 28 finished with value: 5.336767196655273 and parameters: {'n_steps': 256, 'gamma': 0.8, 'learning_rate': 0.0018714248189498342, 'target_kl': 0.1, 'gae_lambda': 1.0, 'batch_size': 8, 'ent_coef': 0.0002}. Best is trial 20 with value: 7.188906669616699.
[I 2025-07-28 13:06:20,726] Trial 29 finished with value: 5.127566337585449 and parameters: {'n_steps': 1024, 'gamma': 0.99, 'learning_rate': 9.83623288027384e-05, 'target_kl': 0.05, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 20 with value: 7.188906669616699.
Completed 10 trials for trpor_with_noise.

Best Trial Stats Across All Configs (Batch 4):
TRPO NO NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 1.0145975538440495e-05, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 1024}
  Best Max Reward: 7.42
TRPO WITH NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.24092835044741195, 'target_kl': 0.03, 'gae_lambda': 0.98, 'batch_size': 16}
  Best Max Reward: 6.52
TRPOR NO NOISE:
  Best Parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 2.0623809231485985e-05, 'target_kl': 0.001, 'gae_lambda': 0.95, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}
  Best Max Reward: 7.16
TRPOR WITH NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 2.3222247001289904e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}
  Best Max Reward: 7.19

Running batch 4 for trpo_no_noise (10 trials for 1000000 timesteps)...
[I 2025-07-28 14:42:43,614] Trial 30 finished with value: 6.646707534790039 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 0.0006829141625502194, 'target_kl': 0.03, 'gae_lambda': 1.0, 'batch_size': 128}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-28 16:09:39,428] Trial 31 finished with value: 6.786592960357666 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 2.3103271525318657e-05, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-28 17:50:03,736] Trial 32 finished with value: 6.8249921798706055 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 2.2120911964217466e-05, 'target_kl': 0.05, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-28 21:11:04,542] Trial 33 finished with value: 5.716462135314941 and parameters: {'n_steps': 64, 'gamma': 0.99, 'learning_rate': 5.3868305898136594e-05, 'target_kl': 0.05, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-29 01:10:52,172] Trial 34 finished with value: 5.510509490966797 and parameters: {'n_steps': 256, 'gamma': 0.99, 'learning_rate': 2.3932563077263073e-05, 'target_kl': 0.05, 'gae_lambda': 0.98, 'batch_size': 32}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-29 02:25:50,021] Trial 35 finished with value: 5.988587379455566 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.14866511606376864, 'target_kl': 0.02, 'gae_lambda': 0.95, 'batch_size': 32}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-29 03:54:08,086] Trial 36 finished with value: 5.480103492736816 and parameters: {'n_steps': 128, 'gamma': 0.9, 'learning_rate': 8.776779274098314e-05, 'target_kl': 0.05, 'gae_lambda': 0.92, 'batch_size': 64}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-29 04:29:16,096] Trial 37 finished with value: 6.010573387145996 and parameters: {'n_steps': 32, 'gamma': 0.99, 'learning_rate': 1.9190173350325662e-05, 'target_kl': 0.1, 'gae_lambda': 0.92, 'batch_size': 256}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-29 07:07:14,762] Trial 38 finished with value: 5.501433849334717 and parameters: {'n_steps': 256, 'gamma': 0.9, 'learning_rate': 0.00015878949212947107, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-29 07:47:31,300] Trial 39 finished with value: 6.570894718170166 and parameters: {'n_steps': 16, 'gamma': 0.8, 'learning_rate': 0.0013142191460581232, 'target_kl': 0.01, 'gae_lambda': 0.92, 'batch_size': 16}. Best is trial 21 with value: 7.419168949127197.
Completed 10 trials for trpo_no_noise.

Running batch 4 for trpo_with_noise (10 trials for 1000000 timesteps)...
[I 2025-07-29 09:01:20,393] Trial 30 finished with value: 6.6256914138793945 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.2892947933773477, 'target_kl': 0.005, 'gae_lambda': 0.8, 'batch_size': 64}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-07-29 10:15:45,575] Trial 31 finished with value: 6.001889705657959 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.3668490542960266, 'target_kl': 0.005, 'gae_lambda': 0.8, 'batch_size': 64}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-07-29 11:30:32,433] Trial 32 finished with value: 6.1356306076049805 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.21571287379023202, 'target_kl': 0.005, 'gae_lambda': 0.8, 'batch_size': 64}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-07-29 12:12:58,642] Trial 33 finished with value: 6.0827741622924805 and parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 0.11337968907003061, 'target_kl': 0.005, 'gae_lambda': 0.95, 'batch_size': 64}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-07-29 13:28:44,371] Trial 34 finished with value: 6.213092803955078 and parameters: {'n_steps': 8, 'gamma': 0.9, 'learning_rate': 0.25710063971089187, 'target_kl': 0.01, 'gae_lambda': 0.8, 'batch_size': 256}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-07-29 14:45:31,430] Trial 35 finished with value: 5.905543327331543 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.036555203371665505, 'target_kl': 0.005, 'gae_lambda': 0.98, 'batch_size': 32}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-07-29 15:12:26,750] Trial 36 finished with value: 6.020667552947998 and parameters: {'n_steps': 32, 'gamma': 0.85, 'learning_rate': 0.019076118848876965, 'target_kl': 0.001, 'gae_lambda': 0.99, 'batch_size': 512}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-07-29 16:31:29,074] Trial 37 finished with value: 6.243753433227539 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.1467314692038394, 'target_kl': 0.01, 'gae_lambda': 0.8, 'batch_size': 128}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-07-29 17:29:57,606] Trial 38 finished with value: 5.304390907287598 and parameters: {'n_steps': 256, 'gamma': 0.99, 'learning_rate': 0.46237781887761725, 'target_kl': 0.05, 'gae_lambda': 0.95, 'batch_size': 1024}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-07-29 18:22:18,502] Trial 39 finished with value: 5.750863075256348 and parameters: {'n_steps': 64, 'gamma': 0.95, 'learning_rate': 0.06295745267885526, 'target_kl': 0.03, 'gae_lambda': 0.99, 'batch_size': 8}. Best is trial 30 with value: 6.6256914138793945.
Completed 10 trials for trpo_with_noise.

Running batch 4 for trpor_no_noise (10 trials for 1000000 timesteps)...
[I 2025-07-29 21:49:31,492] Trial 30 finished with value: 7.111052513122559 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 2.0582981929248426e-05, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0004}. Best is trial 12 with value: 7.1592116355896.
[I 2025-07-30 00:34:59,355] Trial 31 finished with value: 7.451748847961426 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 2.3429274272524663e-05, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
[I 2025-07-30 02:02:23,921] Trial 32 finished with value: 6.645750045776367 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 1.907494241454014e-05, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
[I 2025-07-30 04:25:29,281] Trial 33 finished with value: 5.1893720626831055 and parameters: {'n_steps': 512, 'gamma': 0.99, 'learning_rate': 1.0439007094536265e-05, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0006000000000000001}. Best is trial 31 with value: 7.451748847961426.
[I 2025-07-30 06:03:46,913] Trial 34 finished with value: 6.970202445983887 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 5.4424419623005186e-05, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
[I 2025-07-30 08:54:21,028] Trial 35 finished with value: 5.497927665710449 and parameters: {'n_steps': 128, 'gamma': 0.99, 'learning_rate': 2.2124112956760475e-05, 'target_kl': 0.03, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0006000000000000001}. Best is trial 31 with value: 7.451748847961426.
[I 2025-07-30 10:11:16,592] Trial 36 finished with value: 6.782401084899902 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 2.420062427534825e-05, 'target_kl': 0.05, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0005}. Best is trial 31 with value: 7.451748847961426.
[I 2025-07-30 13:38:25,333] Trial 37 finished with value: 5.124632835388184 and parameters: {'n_steps': 1024, 'gamma': 0.99, 'learning_rate': 0.00012266187999346821, 'target_kl': 0.01, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
[I 2025-07-30 14:26:40,168] Trial 38 finished with value: 5.309598445892334 and parameters: {'n_steps': 512, 'gamma': 0.9, 'learning_rate': 0.012400731141566987, 'target_kl': 0.1, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
[I 2025-07-30 15:54:38,009] Trial 39 finished with value: 5.392556667327881 and parameters: {'n_steps': 256, 'gamma': 0.8, 'learning_rate': 6.23289742822891e-05, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 16, 'ent_coef': 0.0007}. Best is trial 31 with value: 7.451748847961426.
Completed 10 trials for trpor_no_noise.

Running batch 4 for trpor_with_noise (10 trials for 1000000 timesteps)...
[I 2025-07-30 16:57:23,565] Trial 30 finished with value: 5.918369770050049 and parameters: {'n_steps': 64, 'gamma': 0.8, 'learning_rate': 2.5032702419902883e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 64, 'ent_coef': 0.0}. Best is trial 20 with value: 7.188906669616699.
[I 2025-07-30 18:44:42,985] Trial 31 finished with value: 6.797287940979004 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 3.02225097883479e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 20 with value: 7.188906669616699.
[I 2025-07-30 19:30:56,681] Trial 32 finished with value: 5.051971435546875 and parameters: {'n_steps': 2048, 'gamma': 0.8, 'learning_rate': 1.9068604228444206e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.0004}. Best is trial 20 with value: 7.188906669616699.
[I 2025-07-30 21:07:08,906] Trial 33 finished with value: 6.926800727844238 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.00021791143187947857, 'target_kl': 0.001, 'gae_lambda': 0.92, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 20 with value: 7.188906669616699.
[I 2025-07-30 22:42:31,338] Trial 34 finished with value: 6.81626033782959 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 6.702473467830898e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 16, 'ent_coef': 0.0006000000000000001}. Best is trial 20 with value: 7.188906669616699.
[I 2025-07-30 23:23:46,153] Trial 35 finished with value: 5.998932361602783 and parameters: {'n_steps': 32, 'gamma': 0.8, 'learning_rate': 2.154801557491669e-05, 'target_kl': 0.02, 'gae_lambda': 0.98, 'batch_size': 1024, 'ent_coef': 0.0002}. Best is trial 20 with value: 7.188906669616699.
[I 2025-07-31 01:06:22,487] Trial 36 finished with value: 7.273784637451172 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.0003840318016469151, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 32, 'ent_coef': 0.0004}. Best is trial 36 with value: 7.273784637451172.
[I 2025-07-31 03:44:07,748] Trial 37 finished with value: 4.961455821990967 and parameters: {'n_steps': 2048, 'gamma': 0.99, 'learning_rate': 0.007955175486001111, 'target_kl': 0.005, 'gae_lambda': 0.99, 'batch_size': 32, 'ent_coef': 0.0004}. Best is trial 36 with value: 7.273784637451172.
[I 2025-07-31 05:30:00,413] Trial 38 finished with value: 6.543712615966797 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.00033990084298347206, 'target_kl': 0.005, 'gae_lambda': 0.8, 'batch_size': 32, 'ent_coef': 0.0005}. Best is trial 36 with value: 7.273784637451172.
[I 2025-07-31 07:19:04,081] Trial 39 finished with value: 6.733091354370117 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.00012282364290415985, 'target_kl': 0.005, 'gae_lambda': 0.98, 'batch_size': 32, 'ent_coef': 0.0006000000000000001}. Best is trial 36 with value: 7.273784637451172.
Completed 10 trials for trpor_with_noise.

Best Trial Stats Across All Configs (Batch 5):
TRPO NO NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 1.0145975538440495e-05, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 1024}
  Best Max Reward: 7.42
TRPO WITH NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.2892947933773477, 'target_kl': 0.005, 'gae_lambda': 0.8, 'batch_size': 64}
  Best Max Reward: 6.63
TRPOR NO NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 2.3429274272524663e-05, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0004}
  Best Max Reward: 7.45
TRPOR WITH NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.0003840318016469151, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 32, 'ent_coef': 0.0004}
  Best Max Reward: 7.27

Running batch 5 for trpo_no_noise (10 trials for 1000000 timesteps)...
[I 2025-07-31 17:11:38,932] Trial 40 finished with value: 5.352306365966797 and parameters: {'n_steps': 2048, 'gamma': 0.85, 'learning_rate': 6.905781366653996e-05, 'target_kl': 0.03, 'gae_lambda': 0.9, 'batch_size': 128}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-31 18:39:45,652] Trial 41 finished with value: 6.520077228546143 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 1.659054996640434e-05, 'target_kl': 0.05, 'gae_lambda': 0.8, 'batch_size': 8}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-31 20:07:48,485] Trial 42 finished with value: 6.958767890930176 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 1.0422401765917545e-05, 'target_kl': 0.1, 'gae_lambda': 0.8, 'batch_size': 512}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-31 22:11:03,070] Trial 43 finished with value: 6.572930812835693 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 3.3510630645239544e-05, 'target_kl': 0.1, 'gae_lambda': 0.8, 'batch_size': 512}. Best is trial 21 with value: 7.419168949127197.
[I 2025-07-31 23:34:32,694] Trial 44 finished with value: 6.733232021331787 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 3.29178471181229e-05, 'target_kl': 0.1, 'gae_lambda': 0.8, 'batch_size': 512}. Best is trial 21 with value: 7.419168949127197.
[I 2025-08-01 00:37:05,987] Trial 45 finished with value: 5.332150459289551 and parameters: {'n_steps': 512, 'gamma': 0.9, 'learning_rate': 1.792448344120223e-05, 'target_kl': 0.1, 'gae_lambda': 0.98, 'batch_size': 512}. Best is trial 21 with value: 7.419168949127197.
[I 2025-08-01 00:47:14,266] Trial 46 finished with value: 4.816925048828125 and parameters: {'n_steps': 1024, 'gamma': 0.8, 'learning_rate': 0.02221396872979471, 'target_kl': 0.1, 'gae_lambda': 0.8, 'batch_size': 256}. Best is trial 21 with value: 7.419168949127197.
[I 2025-08-01 02:08:48,565] Trial 47 finished with value: 6.7946367263793945 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 1.1167368742471572e-05, 'target_kl': 0.005, 'gae_lambda': 0.95, 'batch_size': 16}. Best is trial 21 with value: 7.419168949127197.
[I 2025-08-01 03:09:29,721] Trial 48 finished with value: 6.147887706756592 and parameters: {'n_steps': 32, 'gamma': 0.95, 'learning_rate': 0.0004223316900879972, 'target_kl': 0.02, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 21 with value: 7.419168949127197.
[I 2025-08-01 04:48:22,672] Trial 49 finished with value: 6.746661186218262 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.0001304514918488745, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 128}. Best is trial 21 with value: 7.419168949127197.
Completed 10 trials for trpo_no_noise.

Running batch 5 for trpo_with_noise (10 trials for 1000000 timesteps)...
[I 2025-08-01 05:32:44,537] Trial 40 finished with value: 6.432667255401611 and parameters: {'n_steps': 16, 'gamma': 0.8, 'learning_rate': 3.4662353152737456e-05, 'target_kl': 0.005, 'gae_lambda': 0.98, 'batch_size': 64}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-08-01 07:17:23,502] Trial 41 finished with value: 6.616805553436279 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.13628133961705982, 'target_kl': 0.03, 'gae_lambda': 0.98, 'batch_size': 16}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-08-01 08:32:06,257] Trial 42 finished with value: 6.092653274536133 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.15766564304915193, 'target_kl': 0.03, 'gae_lambda': 0.98, 'batch_size': 16}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-08-01 09:45:31,216] Trial 43 finished with value: 5.150408744812012 and parameters: {'n_steps': 1024, 'gamma': 0.85, 'learning_rate': 0.000621835224986436, 'target_kl': 0.03, 'gae_lambda': 0.98, 'batch_size': 16}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-08-01 10:06:09,234] Trial 44 finished with value: 4.998343467712402 and parameters: {'n_steps': 2048, 'gamma': 0.85, 'learning_rate': 0.02813942333484363, 'target_kl': 0.001, 'gae_lambda': 0.99, 'batch_size': 16}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-08-01 10:17:29,906] Trial 45 finished with value: 4.898333549499512 and parameters: {'n_steps': 512, 'gamma': 0.85, 'learning_rate': 0.9862291887250613, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 256}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-08-01 11:42:00,282] Trial 46 finished with value: 6.222171783447266 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.5333283689439284, 'target_kl': 0.02, 'gae_lambda': 0.98, 'batch_size': 512}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-08-01 12:07:36,121] Trial 47 finished with value: 5.207248687744141 and parameters: {'n_steps': 128, 'gamma': 0.9, 'learning_rate': 0.08627413427542452, 'target_kl': 0.03, 'gae_lambda': 0.9, 'batch_size': 16}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-08-01 13:30:49,074] Trial 48 finished with value: 6.307526588439941 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.004109549559913402, 'target_kl': 0.01, 'gae_lambda': 1.0, 'batch_size': 16}. Best is trial 30 with value: 6.6256914138793945.
[I 2025-08-01 14:59:10,059] Trial 49 finished with value: 6.880971908569336 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.2733924542207901, 'target_kl': 0.03, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
Completed 10 trials for trpo_with_noise.

Running batch 5 for trpor_no_noise (10 trials for 1000000 timesteps)...
[I 2025-08-01 18:42:22,366] Trial 40 finished with value: 6.093522071838379 and parameters: {'n_steps': 32, 'gamma': 0.99, 'learning_rate': 1.0715517101608945e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0006000000000000001}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-01 20:24:12,273] Trial 41 finished with value: 6.823757648468018 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 5.694187221324855e-05, 'target_kl': 0.005, 'gae_lambda': 0.9, 'batch_size': 256, 'ent_coef': 0.00030000000000000003}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-01 21:47:33,393] Trial 42 finished with value: 6.844424247741699 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.00018362018014735939, 'target_kl': 0.005, 'gae_lambda': 0.8, 'batch_size': 32, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-01 23:24:36,282] Trial 43 finished with value: 6.773748397827148 and parameters: {'n_steps': 8, 'gamma': 0.9, 'learning_rate': 3.4920974399956187e-05, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 64, 'ent_coef': 0.0002}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-02 00:09:59,753] Trial 44 finished with value: 5.573152542114258 and parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 0.10863465144447097, 'target_kl': 0.03, 'gae_lambda': 0.99, 'batch_size': 1024, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-02 00:32:13,351] Trial 45 finished with value: 5.436327934265137 and parameters: {'n_steps': 64, 'gamma': 0.8, 'learning_rate': 1.708852192333725e-05, 'target_kl': 0.1, 'gae_lambda': 0.95, 'batch_size': 128, 'ent_coef': 0.0005}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-02 04:43:32,490] Trial 46 finished with value: 5.039947509765625 and parameters: {'n_steps': 2048, 'gamma': 0.99, 'learning_rate': 8.087145342227174e-05, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 16, 'ent_coef': 0.00030000000000000003}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-02 06:17:03,770] Trial 47 finished with value: 6.751102447509766 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 4.682793268402481e-05, 'target_kl': 0.01, 'gae_lambda': 0.98, 'batch_size': 256, 'ent_coef': 0.0005}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-02 08:33:53,941] Trial 48 finished with value: 6.579534530639648 and parameters: {'n_steps': 16, 'gamma': 0.85, 'learning_rate': 0.00015321352498006097, 'target_kl': 0.005, 'gae_lambda': 0.9, 'batch_size': 512, 'ent_coef': 0.00030000000000000003}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-02 11:50:27,380] Trial 49 finished with value: 6.601297378540039 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 1.5036699902332927e-05, 'target_kl': 0.001, 'gae_lambda': 0.95, 'batch_size': 1024, 'ent_coef': 0.0001}. Best is trial 31 with value: 7.451748847961426.
Completed 10 trials for trpor_no_noise.

Running batch 5 for trpor_with_noise (10 trials for 1000000 timesteps)...
[I 2025-08-02 13:47:10,181] Trial 40 finished with value: 5.249664306640625 and parameters: {'n_steps': 512, 'gamma': 0.9, 'learning_rate': 0.0006068964904962208, 'target_kl': 0.05, 'gae_lambda': 1.0, 'batch_size': 8, 'ent_coef': 0.0002}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-02 15:05:47,598] Trial 41 finished with value: 6.728926658630371 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.0017977853137070671, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 16, 'ent_coef': 0.00030000000000000003}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-02 16:31:18,260] Trial 42 finished with value: 6.7943854331970215 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 4.601751875522209e-05, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.0004}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-02 19:12:47,883] Trial 43 finished with value: 6.89967155456543 and parameters: {'n_steps': 8, 'gamma': 0.9, 'learning_rate': 1.3954817697068885e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 32, 'ent_coef': 0.0005}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-02 20:39:30,052] Trial 44 finished with value: 5.214057922363281 and parameters: {'n_steps': 1024, 'gamma': 0.8, 'learning_rate': 0.0001621749668291078, 'target_kl': 0.1, 'gae_lambda': 0.92, 'batch_size': 64, 'ent_coef': 0.0004}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-03 01:29:12,753] Trial 45 finished with value: 5.438774108886719 and parameters: {'n_steps': 256, 'gamma': 0.99, 'learning_rate': 7.333411645192944e-05, 'target_kl': 0.02, 'gae_lambda': 0.99, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-03 02:35:47,027] Trial 46 finished with value: 5.356868267059326 and parameters: {'n_steps': 128, 'gamma': 0.8, 'learning_rate': 0.03028282127417991, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 512, 'ent_coef': 0.0002}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-03 03:50:46,071] Trial 47 finished with value: 6.264673233032227 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 0.9512073022790389, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.0001}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-03 05:39:47,636] Trial 48 finished with value: 6.126988410949707 and parameters: {'n_steps': 32, 'gamma': 0.9, 'learning_rate': 3.549478537338009e-05, 'target_kl': 0.001, 'gae_lambda': 0.9, 'batch_size': 32, 'ent_coef': 0.00030000000000000003}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-03 07:00:05,754] Trial 49 finished with value: 6.807705402374268 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.28660770035263233, 'target_kl': 0.05, 'gae_lambda': 0.8, 'batch_size': 8, 'ent_coef': 0.0007}. Best is trial 36 with value: 7.273784637451172.
Completed 10 trials for trpor_with_noise.

Best Trial Stats Across All Configs (Batch 6):
TRPO NO NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 1.0145975538440495e-05, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 1024}
  Best Max Reward: 7.42
TRPO WITH NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.2733924542207901, 'target_kl': 0.03, 'gae_lambda': 0.92, 'batch_size': 32}
  Best Max Reward: 6.88
TRPOR NO NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 2.3429274272524663e-05, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0004}
  Best Max Reward: 7.45
TRPOR WITH NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.0003840318016469151, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 32, 'ent_coef': 0.0004}
  Best Max Reward: 7.27

Running batch 6 for trpo_no_noise (10 trials for 1000000 timesteps)...
[I 2025-08-03 10:45:18,386] Trial 50 finished with value: 5.725131034851074 and parameters: {'n_steps': 64, 'gamma': 0.99, 'learning_rate': 4.705434596120773e-05, 'target_kl': 0.03, 'gae_lambda': 0.99, 'batch_size': 512}. Best is trial 21 with value: 7.419168949127197.
[I 2025-08-03 13:20:34,748] Trial 51 finished with value: 6.837060451507568 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 1.4011226219937187e-05, 'target_kl': 0.001, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 21 with value: 7.419168949127197.
^C[W 2025-08-03 14:40:16,662] Trial 52 failed with parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 1.5535144457281555e-05, 'target_kl': 0.001, 'gae_lambda': 0.8, 'batch_size': 64} because of the following error: KeyboardInterrupt().
Traceback (most recent call last):
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/Users/dach/dev/deep-rl/Omega.py", line 357, in <lambda>
    study.optimize(lambda trial: objective(trial, config, env_id, n_timesteps, device, config_to_study), n_trials=current_batch_size)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/dach/dev/deep-rl/Omega.py", line 249, in objective
    model.learn(total_timesteps=n_timesteps, callback=callback)
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/sb3_contrib/trpo/trpo.py", line 414, in learn
    return super().learn(
           ^^^^^^^^^^^^^^
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/stable_baselines3/common/on_policy_algorithm.py", line 336, in learn
    self.train()
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/sb3_contrib/trpo/trpo.py", line 374, in train
    self.policy.optimizer.step()
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/torch/optim/optimizer.py", line 493, in wrapper
    out = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/torch/optim/optimizer.py", line 91, in _use_grad
    ret = func(self, *args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/torch/optim/adam.py", line 244, in step
    adam(
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/torch/optim/optimizer.py", line 154, in maybe_fallback
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/torch/optim/adam.py", line 876, in adam
    func(
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/torch/optim/adam.py", line 476, in _single_tensor_adam
    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
[W 2025-08-03 14:40:16,807] Trial 52 failed with value None.
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/Users/dach/dev/deep-rl/Omega.py", line 403, in <module>
    compare_max_rewards(
  File "/Users/dach/dev/deep-rl/Omega.py", line 357, in compare_max_rewards
    study.optimize(lambda trial: objective(trial, config, env_id, n_timesteps, device, config_to_study), n_trials=current_batch_size)
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/optuna/study/study.py", line 475, in optimize
    _optimize(
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/optuna/study/_optimize.py", line 63, in _optimize
    _optimize_sequential(
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/optuna/study/_optimize.py", line 160, in _optimize_sequential
    frozen_trial = _run_trial(study, func, catch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/optuna/study/_optimize.py", line 248, in _run_trial
    raise func_err
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/optuna/study/_optimize.py", line 197, in _run_trial
    value_or_values = func(trial)
                      ^^^^^^^^^^^
  File "/Users/dach/dev/deep-rl/Omega.py", line 357, in <lambda>
    study.optimize(lambda trial: objective(trial, config, env_id, n_timesteps, device, config_to_study), n_trials=current_batch_size)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/dach/dev/deep-rl/Omega.py", line 249, in objective
    model.learn(total_timesteps=n_timesteps, callback=callback)
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/sb3_contrib/trpo/trpo.py", line 414, in learn
    return super().learn(
           ^^^^^^^^^^^^^^
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/stable_baselines3/common/on_policy_algorithm.py", line 336, in learn
    self.train()
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/sb3_contrib/trpo/trpo.py", line 374, in train
    self.policy.optimizer.step()
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/torch/optim/optimizer.py", line 493, in wrapper
    out = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/torch/optim/optimizer.py", line 91, in _use_grad
    ret = func(self, *args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/torch/optim/adam.py", line 244, in step
    adam(
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/torch/optim/optimizer.py", line 154, in maybe_fallback
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/torch/optim/adam.py", line 876, in adam
    func(
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/torch/optim/adam.py", line 476, in _single_tensor_adam
    denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt
make: *** [omega] Interrupt: 2
dach@dachi ~/d/deep-rl (main) [SIGINT]> make omega
Will run black and isort on modified, added, untracked, or staged Python files
^CTraceback (most recent call last):
  File "/Users/dach/dev/deep-rl/.venv/bin/isort", line 5, in <module>
    from isort.main import main
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/isort/__init__.py", line 23, in <module>
    from ._version import __version__
  File "/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/isort/_version.py", line 1, in <module>
    from importlib import metadata
  File "/opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/lib/python3.12/importlib/metadata/__init__.py", line 19, in <module>
    from . import _adapters, _meta
  File "/opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/lib/python3.12/importlib/metadata/_adapters.py", line 5, in <module>
    import email.message
  File "/opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/lib/python3.12/email/message.py", line 15, in <module>
    from email import utils
  File "/opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/lib/python3.12/email/utils.py", line 28, in <module>
    import random
  File "/opt/homebrew/Cellar/python@3.12/3.12.9/Frameworks/Python.framework/Versions/3.12/lib/python3.12/random.py", line 64, in <module>
    import _random
KeyboardInterrupt
make: *** [fix] Interrupt: 2
dach@dachi ~/d/deep-rl (main) [SIGINT]> killall Python
dach@dachi ~/d/deep-rl (main)> killall Python
dach@dachi ~/d/deep-rl (main)> killall Python
dach@dachi ~/d/deep-rl (main)> killall Python
dach@dachi ~/d/deep-rl (main)> killall Python
dach@dachi ~/d/deep-rl (main)> make omega
Will run black and isort on modified, added, untracked, or staged Python files
All done! ✨ 🍰 ✨
34 files left unchanged.

Setting up study for trpo_no_noise...
[I 2025-08-03 14:40:58,462] Using an existing study with name 'trpo_no_noise_Humanoid-v5_study' instead of creating a new one.

Setting up study for trpo_with_noise...
[I 2025-08-03 14:41:00,256] Using an existing study with name 'trpo_with_noise_Humanoid-v5_study' instead of creating a new one.

Setting up study for trpor_no_noise...
[I 2025-08-03 14:41:01,354] Using an existing study with name 'trpor_no_noise_Humanoid-v5_study' instead of creating a new one.

Setting up study for trpor_with_noise...
[I 2025-08-03 14:41:03,349] Using an existing study with name 'trpor_with_noise_Humanoid-v5_study' instead of creating a new one.

Best Trial Stats Across All Configs (Batch 1):
TRPO NO NOISE: No trials yet
TRPO WITH NOISE: No trials yet
TRPOR NO NOISE: No trials yet
TRPOR WITH NOISE: No trials yet

Running batch 1 for trpo_no_noise (10 trials for 1000000 timesteps)...
/Users/dach/dev/deep-rl/Omega.py:78: UserWarning: Creating legend with loc="best" can be slow with large amounts of data.
  plt.tight_layout()
/Users/dach/dev/deep-rl/Omega.py:79: UserWarning: Creating legend with loc="best" can be slow with large amounts of data.
  plt.savefig(os.path.join(self.log_dir, "graph.png"))
[I 2025-08-03 17:06:33,133] Trial 53 finished with value: 6.8032073974609375 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 1.5198219838643741e-05, 'target_kl': 0.001, 'gae_lambda': 0.8, 'batch_size': 64}. Best is trial 21 with value: 7.419168949127197.
[I 2025-08-03 18:39:40,364] Trial 54 finished with value: 7.503105640411377 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 2.970656446543402e-05, 'target_kl': 0.05, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-03 20:10:50,240] Trial 55 finished with value: 6.758485794067383 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 3.4467324399657866e-05, 'target_kl': 0.001, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-03 20:58:31,162] Trial 56 finished with value: 5.595703601837158 and parameters: {'n_steps': 128, 'gamma': 0.95, 'learning_rate': 1.0174955359304863e-05, 'target_kl': 0.1, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-03 22:29:39,100] Trial 57 finished with value: 5.426095962524414 and parameters: {'n_steps': 2048, 'gamma': 0.95, 'learning_rate': 6.776044041411664e-05, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-03 23:49:16,965] Trial 58 finished with value: 6.473176002502441 and parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 1.669270067999791e-05, 'target_kl': 0.005, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-04 01:42:50,003] Trial 59 finished with value: 6.881682872772217 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 3.036321788187525e-05, 'target_kl': 0.001, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-04 03:04:45,454] Trial 60 finished with value: 6.630728244781494 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.00011041061695667647, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-04 04:19:32,514] Trial 61 finished with value: 6.29075813293457 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.11180604861473255, 'target_kl': 0.001, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-04 06:10:43,731] Trial 62 finished with value: 6.632546424865723 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 2.9997932170452174e-05, 'target_kl': 0.001, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
Completed 10 trials for trpo_no_noise.

Running batch 1 for trpo_with_noise (10 trials for 1000000 timesteps)...
[I 2025-08-04 06:46:07,147] Trial 50 finished with value: 5.205564975738525 and parameters: {'n_steps': 256, 'gamma': 0.99, 'learning_rate': 0.27622451900428013, 'target_kl': 0.1, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-04 08:05:44,169] Trial 51 finished with value: 6.201712131500244 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.16670652286623372, 'target_kl': 0.03, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-04 09:34:32,929] Trial 52 finished with value: 6.540820121765137 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.33970024067765653, 'target_kl': 0.03, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-04 11:03:03,693] Trial 53 finished with value: 6.531803131103516 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.6279549689215981, 'target_kl': 0.03, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-04 12:26:37,145] Trial 54 finished with value: 6.335816383361816 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.6115632019081253, 'target_kl': 0.03, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-04 13:45:36,668] Trial 55 finished with value: 5.878270626068115 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.34885833280842843, 'target_kl': 0.03, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-04 14:09:18,322] Trial 56 finished with value: 5.82918119430542 and parameters: {'n_steps': 64, 'gamma': 0.99, 'learning_rate': 0.9765529001429988, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-04 14:48:20,706] Trial 57 finished with value: 5.201554298400879 and parameters: {'n_steps': 512, 'gamma': 0.99, 'learning_rate': 0.4969575200488934, 'target_kl': 0.03, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-04 16:07:10,589] Trial 58 finished with value: 6.806035041809082 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.25456611935755236, 'target_kl': 0.05, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-04 16:37:48,943] Trial 59 finished with value: 4.999273777008057 and parameters: {'n_steps': 2048, 'gamma': 0.99, 'learning_rate': 0.0918723782561129, 'target_kl': 0.05, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
Completed 10 trials for trpo_with_noise.

Running batch 1 for trpor_no_noise (10 trials for 1000000 timesteps)...
[I 2025-08-04 17:40:12,944] Trial 50 finished with value: 5.739345550537109 and parameters: {'n_steps': 128, 'gamma': 0.95, 'learning_rate': 2.6485128682711257e-05, 'target_kl': 0.02, 'gae_lambda': 0.8, 'batch_size': 256, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/sb3_contrib/trpo/trpo.py:154: UserWarning: You have specified a mini-batch size of 128, but because the `RolloutBuffer` is of size `n_steps * n_envs = 8`, after every 0 untruncated mini-batches, there will be a truncated mini-batch of size 8
We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.
Info: (n_steps=8 and n_envs=1)
  warnings.warn(
[I 2025-08-04 19:09:31,906] Trial 51 finished with value: 6.234555244445801 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.0002439807604236954, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 8, 'ent_coef': 0.00030000000000000003}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-04 21:12:23,465] Trial 52 finished with value: 6.341977119445801 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 7.051896550114627e-05, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 32, 'ent_coef': 0.0002}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-04 22:45:28,426] Trial 53 finished with value: 6.926164627075195 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.0004409621415447278, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 128, 'ent_coef': 0.00030000000000000003}. Best is trial 31 with value: 7.451748847961426.
/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/sb3_contrib/trpo/trpo.py:154: UserWarning: You have specified a mini-batch size of 128, but because the `RolloutBuffer` is of size `n_steps * n_envs = 16`, after every 0 untruncated mini-batches, there will be a truncated mini-batch of size 16
We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.
Info: (n_steps=16 and n_envs=1)
  warnings.warn(
[I 2025-08-04 23:51:56,639] Trial 54 finished with value: 6.21034049987793 and parameters: {'n_steps': 16, 'gamma': 0.99, 'learning_rate': 0.000785700081750414, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 128, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-05 01:10:20,730] Trial 55 finished with value: 6.4356279373168945 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.002630182533969666, 'target_kl': 0.05, 'gae_lambda': 1.0, 'batch_size': 128, 'ent_coef': 0.0002}. Best is trial 31 with value: 7.451748847961426.
/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/sb3_contrib/trpo/trpo.py:154: UserWarning: You have specified a mini-batch size of 128, but because the `RolloutBuffer` is of size `n_steps * n_envs = 32`, after every 0 untruncated mini-batches, there will be a truncated mini-batch of size 32
We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.
Info: (n_steps=32 and n_envs=1)
  warnings.warn(
[I 2025-08-05 02:34:06,699] Trial 56 finished with value: 6.405397891998291 and parameters: {'n_steps': 32, 'gamma': 0.95, 'learning_rate': 3.190637274489215e-05, 'target_kl': 0.005, 'gae_lambda': 0.95, 'batch_size': 128, 'ent_coef': 0.0005}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-05 03:39:55,244] Trial 57 finished with value: 5.036293029785156 and parameters: {'n_steps': 2048, 'gamma': 0.99, 'learning_rate': 0.00012068282449915236, 'target_kl': 0.001, 'gae_lambda': 0.92, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-05 05:07:15,072] Trial 58 finished with value: 6.622805595397949 and parameters: {'n_steps': 8, 'gamma': 0.9, 'learning_rate': 4.691541484697031e-05, 'target_kl': 0.005, 'gae_lambda': 0.99, 'batch_size': 64, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
/Users/dach/dev/deep-rl/.venv/lib/python3.12/site-packages/sb3_contrib/trpo/trpo.py:154: UserWarning: You have specified a mini-batch size of 128, but because the `RolloutBuffer` is of size `n_steps * n_envs = 64`, after every 0 untruncated mini-batches, there will be a truncated mini-batch of size 64
We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.
Info: (n_steps=64 and n_envs=1)
  warnings.warn(
[I 2025-08-05 05:46:36,207] Trial 59 finished with value: 5.72181510925293 and parameters: {'n_steps': 64, 'gamma': 0.95, 'learning_rate': 0.00044946978950999853, 'target_kl': 0.03, 'gae_lambda': 1.0, 'batch_size': 128, 'ent_coef': 0.001}. Best is trial 31 with value: 7.451748847961426.
Completed 10 trials for trpor_no_noise.

Running batch 1 for trpor_with_noise (10 trials for 1000000 timesteps)...
[I 2025-08-05 06:08:12,301] Trial 50 finished with value: 5.767300605773926 and parameters: {'n_steps': 64, 'gamma': 0.8, 'learning_rate': 0.00039763037853882843, 'target_kl': 0.1, 'gae_lambda': 0.98, 'batch_size': 128, 'ent_coef': 0.0004}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-05 07:41:44,139] Trial 51 finished with value: 6.686069011688232 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.00024512335213306596, 'target_kl': 0.001, 'gae_lambda': 0.92, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-05 09:30:40,904] Trial 52 finished with value: 6.484296798706055 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 8.278469113325214e-05, 'target_kl': 0.001, 'gae_lambda': 0.92, 'batch_size': 1024, 'ent_coef': 0.0002}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-05 10:53:16,880] Trial 53 finished with value: 6.857011318206787 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.000181238656684041, 'target_kl': 0.001, 'gae_lambda': 0.92, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-05 12:10:31,582] Trial 54 finished with value: 6.849735260009766 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.001606529208871572, 'target_kl': 0.001, 'gae_lambda': 0.92, 'batch_size': 1024, 'ent_coef': 0.0005}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-05 13:25:24,223] Trial 55 finished with value: 6.652926445007324 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.004170463018000091, 'target_kl': 0.01, 'gae_lambda': 0.95, 'batch_size': 1024, 'ent_coef': 0.0004}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-05 14:56:49,450] Trial 56 finished with value: 5.702984809875488 and parameters: {'n_steps': 128, 'gamma': 0.95, 'learning_rate': 0.0006846854731309375, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 512, 'ent_coef': 0.0001}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-05 16:51:14,234] Trial 57 finished with value: 5.026809215545654 and parameters: {'n_steps': 2048, 'gamma': 0.8, 'learning_rate': 0.0003152077385599081, 'target_kl': 0.02, 'gae_lambda': 1.0, 'batch_size': 16, 'ent_coef': 0.0002}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-05 18:07:33,584] Trial 58 finished with value: 7.037075042724609 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 5.709515759636933e-05, 'target_kl': 0.001, 'gae_lambda': 0.9, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-05 19:05:29,249] Trial 59 finished with value: 5.321834564208984 and parameters: {'n_steps': 512, 'gamma': 0.8, 'learning_rate': 1.3849551636442145e-05, 'target_kl': 0.005, 'gae_lambda': 0.9, 'batch_size': 64, 'ent_coef': 0.0004}. Best is trial 36 with value: 7.273784637451172.
Completed 10 trials for trpor_with_noise.

Best Trial Stats Across All Configs (Batch 2):
TRPO NO NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 2.970656446543402e-05, 'target_kl': 0.05, 'gae_lambda': 0.8, 'batch_size': 1024}
  Best Max Reward: 7.50
TRPO WITH NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.2733924542207901, 'target_kl': 0.03, 'gae_lambda': 0.92, 'batch_size': 32}
  Best Max Reward: 6.88
TRPOR NO NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 2.3429274272524663e-05, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0004}
  Best Max Reward: 7.45
TRPOR WITH NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.0003840318016469151, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 32, 'ent_coef': 0.0004}
  Best Max Reward: 7.27

Running batch 2 for trpo_no_noise (10 trials for 1000000 timesteps)...
[I 2025-08-05 19:53:08,949] Trial 63 finished with value: 5.32954216003418 and parameters: {'n_steps': 1024, 'gamma': 0.95, 'learning_rate': 4.432459124850107e-05, 'target_kl': 0.001, 'gae_lambda': 0.8, 'batch_size': 128}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-05 21:14:34,144] Trial 64 finished with value: 6.737534523010254 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.9667098751980294, 'target_kl': 0.03, 'gae_lambda': 0.8, 'batch_size': 8}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-05 22:07:53,368] Trial 65 finished with value: 5.206403732299805 and parameters: {'n_steps': 512, 'gamma': 0.95, 'learning_rate': 1.344689974314753e-05, 'target_kl': 0.01, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-05 23:32:39,111] Trial 66 finished with value: 6.703337669372559 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 2.7317329079322118e-05, 'target_kl': 0.03, 'gae_lambda': 0.9, 'batch_size': 128}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-06 00:22:40,614] Trial 67 finished with value: 6.458044052124023 and parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 1.0020756590379684e-05, 'target_kl': 0.05, 'gae_lambda': 0.8, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-06 01:39:35,087] Trial 68 finished with value: 5.589323043823242 and parameters: {'n_steps': 256, 'gamma': 0.95, 'learning_rate': 6.254454366678853e-05, 'target_kl': 0.02, 'gae_lambda': 0.8, 'batch_size': 512}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-06 02:56:19,080] Trial 69 finished with value: 6.988149166107178 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 1.570476236591467e-05, 'target_kl': 0.001, 'gae_lambda': 0.99, 'batch_size': 256}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-06 04:14:33,354] Trial 70 finished with value: 6.694953918457031 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.0001832803911909475, 'target_kl': 0.001, 'gae_lambda': 0.99, 'batch_size': 256}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-06 05:28:11,121] Trial 71 finished with value: 6.354098320007324 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.007896455804177327, 'target_kl': 0.001, 'gae_lambda': 0.99, 'batch_size': 256}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-06 06:56:32,322] Trial 72 finished with value: 6.768033027648926 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 2.265030943756896e-05, 'target_kl': 0.001, 'gae_lambda': 0.99, 'batch_size': 256}. Best is trial 54 with value: 7.503105640411377.
Completed 10 trials for trpo_no_noise.

Running batch 2 for trpo_with_noise (10 trials for 1000000 timesteps)...
[I 2025-08-06 07:36:53,494] Trial 60 finished with value: 4.982273578643799 and parameters: {'n_steps': 1024, 'gamma': 0.99, 'learning_rate': 0.67098444264179, 'target_kl': 0.05, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-06 08:51:39,206] Trial 61 finished with value: 5.854414939880371 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.27548592175074454, 'target_kl': 0.05, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-06 10:06:39,804] Trial 62 finished with value: 6.787611961364746 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.3898592294287282, 'target_kl': 0.05, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-06 11:20:46,356] Trial 63 finished with value: 6.793004035949707 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.37660963146956034, 'target_kl': 0.05, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-06 12:36:45,018] Trial 64 finished with value: 6.33636999130249 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.39799027622371325, 'target_kl': 0.05, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-06 13:50:32,557] Trial 65 finished with value: 6.210284233093262 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.11985701056759028, 'target_kl': 0.05, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-06 14:16:11,798] Trial 66 finished with value: 5.7975921630859375 and parameters: {'n_steps': 32, 'gamma': 0.99, 'learning_rate': 0.2150676475702079, 'target_kl': 0.05, 'gae_lambda': 0.92, 'batch_size': 128}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-06 14:44:05,910] Trial 67 finished with value: 5.380100727081299 and parameters: {'n_steps': 128, 'gamma': 0.99, 'learning_rate': 0.07024677035530989, 'target_kl': 0.05, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-06 15:34:34,664] Trial 68 finished with value: 5.634676933288574 and parameters: {'n_steps': 16, 'gamma': 0.8, 'learning_rate': 0.00037902428314044766, 'target_kl': 0.05, 'gae_lambda': 0.92, 'batch_size': 8}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-06 16:48:55,319] Trial 69 finished with value: 6.512842655181885 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.18379011984681706, 'target_kl': 0.05, 'gae_lambda': 0.95, 'batch_size': 64}. Best is trial 49 with value: 6.880971908569336.
Completed 10 trials for trpo_with_noise.

Running batch 2 for trpor_no_noise (10 trials for 1000000 timesteps)...
[I 2025-08-06 17:08:49,474] Trial 60 finished with value: 5.2773213386535645 and parameters: {'n_steps': 256, 'gamma': 0.8, 'learning_rate': 0.00920079835501876, 'target_kl': 0.1, 'gae_lambda': 0.98, 'batch_size': 256, 'ent_coef': 0.00030000000000000003}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-06 18:24:30,230] Trial 61 finished with value: 6.675589561462402 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.0019070164316263049, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 8, 'ent_coef': 0.00030000000000000003}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-06 20:44:28,616] Trial 62 finished with value: 6.653036117553711 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 1.4983848846297331e-05, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 8, 'ent_coef': 0.0002}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-06 22:29:29,917] Trial 63 finished with value: 6.560135841369629 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 9.647177603275843e-05, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 512, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-07 00:36:05,756] Trial 64 finished with value: 5.154381275177002 and parameters: {'n_steps': 1024, 'gamma': 0.99, 'learning_rate': 0.0003614881373661009, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 16, 'ent_coef': 0.00030000000000000003}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-07 02:32:26,690] Trial 65 finished with value: 6.770869255065918 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 2.7498796491851115e-05, 'target_kl': 0.005, 'gae_lambda': 0.95, 'batch_size': 8, 'ent_coef': 0.0005}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-07 04:12:22,473] Trial 66 finished with value: 5.4638471603393555 and parameters: {'n_steps': 512, 'gamma': 0.95, 'learning_rate': 0.0006551547554160177, 'target_kl': 0.02, 'gae_lambda': 0.92, 'batch_size': 256, 'ent_coef': 0.00030000000000000003}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-07 05:45:00,470] Trial 67 finished with value: 6.248761177062988 and parameters: {'n_steps': 16, 'gamma': 0.99, 'learning_rate': 0.00019772182226995375, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.0001}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-07 07:16:03,760] Trial 68 finished with value: 6.72499942779541 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 4.366232152075555e-05, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 8, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-07 08:36:53,285] Trial 69 finished with value: 6.848523139953613 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 1.2186484533787631e-05, 'target_kl': 0.01, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0002}. Best is trial 31 with value: 7.451748847961426.
Completed 10 trials for trpor_no_noise.

Running batch 2 for trpor_with_noise (10 trials for 1000000 timesteps)...
[I 2025-08-07 09:39:30,315] Trial 60 finished with value: 6.665063858032227 and parameters: {'n_steps': 16, 'gamma': 0.8, 'learning_rate': 5.9553321448028843e-05, 'target_kl': 0.001, 'gae_lambda': 0.9, 'batch_size': 1024, 'ent_coef': 0.0005}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-07 11:02:55,442] Trial 61 finished with value: 6.579241752624512 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 0.0001354953900678771, 'target_kl': 0.001, 'gae_lambda': 0.9, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-07 12:30:19,714] Trial 62 finished with value: 6.800004959106445 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 3.0977396822885476e-05, 'target_kl': 0.001, 'gae_lambda': 0.9, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-07 13:50:09,926] Trial 63 finished with value: 6.757627487182617 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 4.306101070649436e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.00030000000000000003}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-07 14:53:09,510] Trial 64 finished with value: 5.210552215576172 and parameters: {'n_steps': 1024, 'gamma': 0.95, 'learning_rate': 9.419169184363022e-05, 'target_kl': 0.001, 'gae_lambda': 0.99, 'batch_size': 1024, 'ent_coef': 0.0002}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-07 16:21:42,527] Trial 65 finished with value: 6.797170639038086 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 1.5787361186322915e-05, 'target_kl': 0.01, 'gae_lambda': 1.0, 'batch_size': 32, 'ent_coef': 0.0004}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-07 17:50:33,614] Trial 66 finished with value: 6.513677597045898 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 1.0490244130616874e-05, 'target_kl': 0.03, 'gae_lambda': 0.92, 'batch_size': 1024, 'ent_coef': 0.0004}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-07 18:16:37,606] Trial 67 finished with value: 5.391354560852051 and parameters: {'n_steps': 256, 'gamma': 0.8, 'learning_rate': 2.6242350237518323e-05, 'target_kl': 0.001, 'gae_lambda': 0.95, 'batch_size': 128, 'ent_coef': 0.00030000000000000003}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-07 20:23:12,014] Trial 68 finished with value: 6.231606483459473 and parameters: {'n_steps': 32, 'gamma': 0.99, 'learning_rate': 6.141052241647138e-05, 'target_kl': 0.001, 'gae_lambda': 0.8, 'batch_size': 1024, 'ent_coef': 0.0005}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-07 20:59:32,361] Trial 69 finished with value: 5.796217918395996 and parameters: {'n_steps': 64, 'gamma': 0.8, 'learning_rate': 0.0001185821510724308, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0006000000000000001}. Best is trial 36 with value: 7.273784637451172.
Completed 10 trials for trpor_with_noise.

Best Trial Stats Across All Configs (Batch 3):
TRPO NO NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 2.970656446543402e-05, 'target_kl': 0.05, 'gae_lambda': 0.8, 'batch_size': 1024}
  Best Max Reward: 7.50
TRPO WITH NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.2733924542207901, 'target_kl': 0.03, 'gae_lambda': 0.92, 'batch_size': 32}
  Best Max Reward: 6.88
TRPOR NO NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 2.3429274272524663e-05, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0004}
  Best Max Reward: 7.45
TRPOR WITH NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.0003840318016469151, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 32, 'ent_coef': 0.0004}
  Best Max Reward: 7.27

Running batch 3 for trpo_no_noise (10 trials for 1000000 timesteps)...
[I 2025-08-07 22:26:03,705] Trial 73 finished with value: 6.923727512359619 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 1.4231648935489808e-05, 'target_kl': 0.001, 'gae_lambda': 0.98, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-07 23:49:55,724] Trial 74 finished with value: 6.934074401855469 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 1.4623450681611534e-05, 'target_kl': 0.001, 'gae_lambda': 0.98, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-08 01:14:06,939] Trial 75 finished with value: 6.834300994873047 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 4.4394040886121395e-05, 'target_kl': 0.001, 'gae_lambda': 0.98, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-08 02:32:42,426] Trial 76 finished with value: 7.222168922424316 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 2.0235648521336685e-05, 'target_kl': 0.001, 'gae_lambda': 0.98, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-08 03:47:21,552] Trial 77 finished with value: 6.675889492034912 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 1.5140037215617572e-05, 'target_kl': 0.001, 'gae_lambda': 0.98, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-08 05:09:25,054] Trial 78 finished with value: 6.657632827758789 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 2.3444968055428985e-05, 'target_kl': 0.001, 'gae_lambda': 0.98, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-08 05:36:11,013] Trial 79 finished with value: 6.109438896179199 and parameters: {'n_steps': 32, 'gamma': 0.85, 'learning_rate': 1.866930142666148e-05, 'target_kl': 0.001, 'gae_lambda': 0.98, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-08 06:51:18,458] Trial 80 finished with value: 6.784812927246094 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 3.8385448049695584e-05, 'target_kl': 0.001, 'gae_lambda': 0.98, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-08 07:45:06,182] Trial 81 finished with value: 5.570856094360352 and parameters: {'n_steps': 64, 'gamma': 0.85, 'learning_rate': 2.7941560731118553e-05, 'target_kl': 0.05, 'gae_lambda': 0.98, 'batch_size': 16}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-08 09:02:11,761] Trial 82 finished with value: 6.815760612487793 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 1.2248532794855849e-05, 'target_kl': 0.001, 'gae_lambda': 0.98, 'batch_size': 64}. Best is trial 54 with value: 7.503105640411377.
Completed 10 trials for trpo_no_noise.

Running batch 3 for trpo_with_noise (10 trials for 1000000 timesteps)...
[I 2025-08-08 10:16:14,172] Trial 70 finished with value: 6.045071601867676 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.3514346649244316, 'target_kl': 0.02, 'gae_lambda': 0.8, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-08 11:29:42,437] Trial 71 finished with value: 6.533473014831543 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.4472973178833117, 'target_kl': 0.001, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-08 12:42:56,450] Trial 72 finished with value: 5.879947662353516 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.4302703711025752, 'target_kl': 0.001, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-08 13:57:00,724] Trial 73 finished with value: 6.291898727416992 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.744788670802196, 'target_kl': 0.001, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-08 15:10:39,069] Trial 74 finished with value: 6.548720359802246 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.11884623948569757, 'target_kl': 0.001, 'gae_lambda': 0.92, 'batch_size': 32}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-08 16:32:58,419] Trial 75 finished with value: 6.64585542678833 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.21451589220896922, 'target_kl': 0.1, 'gae_lambda': 0.9, 'batch_size': 1024}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-08 18:05:55,718] Trial 76 finished with value: 6.482425212860107 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.14330669144374417, 'target_kl': 0.1, 'gae_lambda': 0.9, 'batch_size': 1024}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-08 18:44:25,772] Trial 77 finished with value: 5.187299728393555 and parameters: {'n_steps': 256, 'gamma': 0.95, 'learning_rate': 0.1095251404072344, 'target_kl': 0.1, 'gae_lambda': 0.9, 'batch_size': 1024}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-08 20:10:46,190] Trial 78 finished with value: 6.573103904724121 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.027363083633844352, 'target_kl': 0.1, 'gae_lambda': 0.9, 'batch_size': 1024}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-08 20:31:00,841] Trial 79 finished with value: 5.673920154571533 and parameters: {'n_steps': 64, 'gamma': 0.95, 'learning_rate': 0.011357433180536756, 'target_kl': 0.1, 'gae_lambda': 0.9, 'batch_size': 1024}. Best is trial 49 with value: 6.880971908569336.
Completed 10 trials for trpo_with_noise.

Running batch 3 for trpor_no_noise (10 trials for 1000000 timesteps)...
[I 2025-08-08 21:55:17,042] Trial 70 finished with value: 6.375197410583496 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.0014557290197845499, 'target_kl': 0.05, 'gae_lambda': 0.9, 'batch_size': 32, 'ent_coef': 0.0002}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-08 23:35:39,656] Trial 71 finished with value: 7.17281436920166 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.00030589682660449633, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 8, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-09 01:10:22,528] Trial 72 finished with value: 6.420506477355957 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.0006612704964914343, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 8, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-09 03:03:50,929] Trial 73 finished with value: 6.641727447509766 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 2.1932129428134527e-05, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 8, 'ent_coef': 0.00030000000000000003}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-09 04:24:02,059] Trial 74 finished with value: 6.961407661437988 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 6.364237983136825e-05, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 8, 'ent_coef': 0.0005}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-09 06:10:54,845] Trial 75 finished with value: 6.811081409454346 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 6.171133346261159e-05, 'target_kl': 0.005, 'gae_lambda': 0.8, 'batch_size': 8, 'ent_coef': 0.0005}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-09 08:45:37,392] Trial 76 finished with value: 5.531591415405273 and parameters: {'n_steps': 128, 'gamma': 0.99, 'learning_rate': 9.342222592218653e-05, 'target_kl': 0.005, 'gae_lambda': 0.95, 'batch_size': 256, 'ent_coef': 0.0006000000000000001}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-09 09:25:44,942] Trial 77 finished with value: 6.281806468963623 and parameters: {'n_steps': 32, 'gamma': 0.9, 'learning_rate': 0.00015555629971357514, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 128, 'ent_coef': 0.0005}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-09 10:35:06,362] Trial 78 finished with value: 6.146167755126953 and parameters: {'n_steps': 16, 'gamma': 0.99, 'learning_rate': 3.3132688552614137e-05, 'target_kl': 0.03, 'gae_lambda': 1.0, 'batch_size': 64, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-09 12:34:48,790] Trial 79 finished with value: 5.509123802185059 and parameters: {'n_steps': 256, 'gamma': 0.95, 'learning_rate': 5.5052128234181636e-05, 'target_kl': 0.001, 'gae_lambda': 0.99, 'batch_size': 1024, 'ent_coef': 0.0005}. Best is trial 31 with value: 7.451748847961426.
Completed 10 trials for trpor_no_noise.

Running batch 3 for trpor_with_noise (10 trials for 1000000 timesteps)...
[I 2025-08-09 13:55:25,605] Trial 70 finished with value: 6.519992828369141 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 1.889812817670667e-05, 'target_kl': 0.05, 'gae_lambda': 1.0, 'batch_size': 1024, 'ent_coef': 0.0002}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-09 15:23:28,637] Trial 71 finished with value: 6.810820579528809 and parameters: {'n_steps': 8, 'gamma': 0.9, 'learning_rate': 1.3093298472242235e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 32, 'ent_coef': 0.0005}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-09 16:55:26,463] Trial 72 finished with value: 6.802141189575195 and parameters: {'n_steps': 8, 'gamma': 0.9, 'learning_rate': 3.642422589663119e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 32, 'ent_coef': 0.0004}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-09 18:22:15,327] Trial 73 finished with value: 6.625541687011719 and parameters: {'n_steps': 8, 'gamma': 0.9, 'learning_rate': 2.0191217587152577e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 32, 'ent_coef': 0.0005}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-09 19:48:53,183] Trial 74 finished with value: 6.939735412597656 and parameters: {'n_steps': 8, 'gamma': 0.9, 'learning_rate': 5.158056617802664e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 32, 'ent_coef': 0.0006000000000000001}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-09 21:37:12,055] Trial 75 finished with value: 6.834075450897217 and parameters: {'n_steps': 8, 'gamma': 0.9, 'learning_rate': 5.509622019800133e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 8, 'ent_coef': 0.0006000000000000001}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-09 22:59:26,454] Trial 76 finished with value: 6.617929935455322 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.00024030755504424515, 'target_kl': 0.1, 'gae_lambda': 0.98, 'batch_size': 32, 'ent_coef': 0.0007}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-09 23:39:03,934] Trial 77 finished with value: 4.985604763031006 and parameters: {'n_steps': 2048, 'gamma': 0.8, 'learning_rate': 0.00016323088521922157, 'target_kl': 0.001, 'gae_lambda': 0.9, 'batch_size': 1024, 'ent_coef': 0.0004}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-10 01:06:36,534] Trial 78 finished with value: 5.615270137786865 and parameters: {'n_steps': 128, 'gamma': 0.9, 'learning_rate': 0.0004871362998068778, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 16, 'ent_coef': 0.0008}. Best is trial 36 with value: 7.273784637451172.
[I 2025-08-10 01:59:52,034] Trial 79 finished with value: 6.388805389404297 and parameters: {'n_steps': 16, 'gamma': 0.95, 'learning_rate': 8.184736224310486e-05, 'target_kl': 0.02, 'gae_lambda': 1.0, 'batch_size': 512, 'ent_coef': 0.00030000000000000003}. Best is trial 36 with value: 7.273784637451172.
Completed 10 trials for trpor_with_noise.

Best Trial Stats Across All Configs (Batch 4):
TRPO NO NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 2.970656446543402e-05, 'target_kl': 0.05, 'gae_lambda': 0.8, 'batch_size': 1024}
  Best Max Reward: 7.50
TRPO WITH NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.2733924542207901, 'target_kl': 0.03, 'gae_lambda': 0.92, 'batch_size': 32}
  Best Max Reward: 6.88
TRPOR NO NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 2.3429274272524663e-05, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.0004}
  Best Max Reward: 7.45
TRPOR WITH NOISE:
  Best Parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.0003840318016469151, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 32, 'ent_coef': 0.0004}
  Best Max Reward: 7.27

Running batch 4 for trpo_no_noise (10 trials for 1000000 timesteps)...
[I 2025-08-10 03:15:14,576] Trial 83 finished with value: 6.642021179199219 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 8.604267224994609e-05, 'target_kl': 0.1, 'gae_lambda': 0.98, 'batch_size': 256}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-10 04:32:25,575] Trial 84 finished with value: 6.792160987854004 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 1.7931595300690055e-05, 'target_kl': 0.01, 'gae_lambda': 0.98, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-10 05:48:28,872] Trial 85 finished with value: 6.606590747833252 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 2.389287606675056e-05, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 512}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-10 06:32:00,870] Trial 86 finished with value: 5.435554504394531 and parameters: {'n_steps': 128, 'gamma': 0.9, 'learning_rate': 1.2867930615762124e-05, 'target_kl': 0.05, 'gae_lambda': 0.95, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-10 06:58:44,401] Trial 87 finished with value: 5.10880708694458 and parameters: {'n_steps': 2048, 'gamma': 0.85, 'learning_rate': 5.398012242236643e-05, 'target_kl': 0.001, 'gae_lambda': 0.99, 'batch_size': 8}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-10 08:13:52,359] Trial 88 finished with value: 6.3097333908081055 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.002571881279906604, 'target_kl': 0.1, 'gae_lambda': 0.98, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-10 09:31:28,441] Trial 89 finished with value: 6.88865327835083 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 3.78741781957328e-05, 'target_kl': 0.005, 'gae_lambda': 0.9, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-10 09:57:39,206] Trial 90 finished with value: 5.432596683502197 and parameters: {'n_steps': 512, 'gamma': 0.85, 'learning_rate': 3.382191450211041e-05, 'target_kl': 0.005, 'gae_lambda': 0.9, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-10 10:11:02,466] Trial 91 finished with value: 5.228034496307373 and parameters: {'n_steps': 1024, 'gamma': 0.8, 'learning_rate': 1.0163715098964402e-05, 'target_kl': 0.005, 'gae_lambda': 0.98, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
[I 2025-08-10 11:29:49,538] Trial 92 finished with value: 6.841909408569336 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 2.0825025054921128e-05, 'target_kl': 0.005, 'gae_lambda': 0.9, 'batch_size': 1024}. Best is trial 54 with value: 7.503105640411377.
Completed 10 trials for trpo_no_noise.

Running batch 4 for trpo_with_noise (10 trials for 1000000 timesteps)...
[I 2025-08-10 12:45:28,541] Trial 80 finished with value: 6.041656970977783 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.027428309612009826, 'target_kl': 0.1, 'gae_lambda': 0.9, 'batch_size': 1024}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-10 14:04:20,621] Trial 81 finished with value: 6.50194787979126 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.05032605855408764, 'target_kl': 0.1, 'gae_lambda': 0.9, 'batch_size': 1024}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-10 15:20:53,845] Trial 82 finished with value: 6.388404846191406 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.19819399720837316, 'target_kl': 0.1, 'gae_lambda': 0.9, 'batch_size': 1024}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-10 16:41:39,146] Trial 83 finished with value: 6.197932243347168 and parameters: {'n_steps': 8, 'gamma': 0.9, 'learning_rate': 0.06810794271557405, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 256}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-10 17:10:32,249] Trial 84 finished with value: 5.311580657958984 and parameters: {'n_steps': 32, 'gamma': 0.95, 'learning_rate': 0.2575342226809423, 'target_kl': 0.001, 'gae_lambda': 0.8, 'batch_size': 64}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-10 18:38:53,606] Trial 85 finished with value: 6.652308464050293 and parameters: {'n_steps': 8, 'gamma': 0.95, 'learning_rate': 0.013017297013806826, 'target_kl': 0.05, 'gae_lambda': 0.9, 'batch_size': 64}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-10 19:11:11,865] Trial 86 finished with value: 5.096364974975586 and parameters: {'n_steps': 2048, 'gamma': 0.95, 'learning_rate': 0.02351276977870549, 'target_kl': 0.05, 'gae_lambda': 0.9, 'batch_size': 64}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-10 19:57:55,472] Trial 87 finished with value: 5.153720378875732 and parameters: {'n_steps': 512, 'gamma': 0.95, 'learning_rate': 0.003117122387157433, 'target_kl': 0.05, 'gae_lambda': 0.9, 'batch_size': 64}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-10 20:39:04,409] Trial 88 finished with value: 5.0496931076049805 and parameters: {'n_steps': 1024, 'gamma': 0.95, 'learning_rate': 0.014664284109191891, 'target_kl': 0.05, 'gae_lambda': 0.9, 'batch_size': 512}. Best is trial 49 with value: 6.880971908569336.
[I 2025-08-10 22:12:27,939] Trial 89 finished with value: 6.768409729003906 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 0.005320534277351914, 'target_kl': 0.01, 'gae_lambda': 0.9, 'batch_size': 64}. Best is trial 49 with value: 6.880971908569336.
Completed 10 trials for trpo_with_noise.

Running batch 4 for trpor_no_noise (10 trials for 1000000 timesteps)...
[I 2025-08-10 23:28:22,585] Trial 80 finished with value: 5.214706897735596 and parameters: {'n_steps': 1024, 'gamma': 0.8, 'learning_rate': 1.7608336899089716e-05, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 512, 'ent_coef': 0.0009000000000000001}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-11 00:50:24,762] Trial 81 finished with value: 6.803333282470703 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.00034657053078677004, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 8, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-11 03:45:11,498] Trial 82 finished with value: 6.894041538238525 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 1.016472064776797e-05, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 8, 'ent_coef': 0.00030000000000000003}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-11 06:16:39,268] Trial 83 finished with value: 6.598015785217285 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 7.83270370363325e-05, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 8, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-11 08:41:18,324] Trial 84 finished with value: 6.38935661315918 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.00022199805731163612, 'target_kl': 0.005, 'gae_lambda': 1.0, 'batch_size': 256, 'ent_coef': 0.00030000000000000003}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-11 10:25:59,401] Trial 85 finished with value: 5.033579349517822 and parameters: {'n_steps': 512, 'gamma': 0.99, 'learning_rate': 0.0009249095472100455, 'target_kl': 0.1, 'gae_lambda': 0.98, 'batch_size': 8, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-11 13:45:45,969] Trial 86 finished with value: 5.2490434646606445 and parameters: {'n_steps': 2048, 'gamma': 0.95, 'learning_rate': 4.263500708122712e-05, 'target_kl': 0.02, 'gae_lambda': 0.92, 'batch_size': 16, 'ent_coef': 0.0006000000000000001}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-11 15:19:18,387] Trial 87 finished with value: 6.503749847412109 and parameters: {'n_steps': 8, 'gamma': 0.85, 'learning_rate': 0.00012596048916913862, 'target_kl': 0.005, 'gae_lambda': 0.95, 'batch_size': 256, 'ent_coef': 0.0005}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-11 17:02:37,606] Trial 88 finished with value: 7.190394878387451 and parameters: {'n_steps': 8, 'gamma': 0.99, 'learning_rate': 0.00047272143116101624, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 8, 'ent_coef': 0.0004}. Best is trial 31 with value: 7.451748847961426.
[I 2025-08-11 18:49:53,603] Trial 89 finished with value: 5.616150379180908 and parameters: {'n_steps': 64, 'gamma': 0.99, 'learning_rate': 0.0005307809754626271, 'target_kl': 0.001, 'gae_lambda': 1.0, 'batch_size': 8, 'ent_coef': 0.0005}. Best is trial 31 with value: 7.451748847961426.
Completed 10 trials for trpor_no_noise.

Running batch 4 for trpor_with_noise (10 trials for 1000000 timesteps)...
[I 2025-08-11 20:29:02,772] Trial 80 finished with value: 6.8264288902282715 and parameters: {'n_steps': 8, 'gamma': 0.8, 'learning_rate': 4.106334525229211e-05, 'target_kl': 0.005, 'gae_lambda': 0.92, 'batch_size': 64, 'ent_coef': 0.0006000000000000001}. Best is trial 36 with value: 7.273784637451172.
