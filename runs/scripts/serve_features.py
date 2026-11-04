"""serve_seeded.py's GR00T N1.5 policy server that also returns the action head's features of the sampled chunk.

Same policy, same deterministic torch settings and the same "_seed" handling (the seeded endpoint is wrapped, not
rewritten). Added: a forward hook on the action head's DiT keeps the output of every DiT call in a request, and one extra
DiT call after the sampling ("clean pass"): the finished chunk at timestep bucket num_timestep_buckets - 1 (t = 0.999,
the cleanest the head was trained on; t = 1 is the action itself), with the same state and vision-language tokens. The
extra call runs after the chunk is sampled and draws no random numbers, so the returned action is unchanged. Returned:
  _feat        the last denoising step's DiT output at the action positions (action_horizon x 1024, float32);
  _feat_clean  the clean pass's DiT output at the action positions (action_horizon x 1024);
  _feat_steps  every denoising step's output, averaged over the action positions (num_inference_timesteps x 1024);
  _n_dit       the number of DiT calls.
A request raises unless the DiT ran exactly num_inference_timesteps + 1 (5) times, so a hook or wrapper that is off the
call path cannot pass silently.
Why these features: DiVeR (arXiv 2610.04933) represents a candidate chunk by the action expert's final-layer action
tokens at the final denoising step. GR00T's counterpart is _feat, the DiT output at the action positions, which the action
decoder maps to the velocity (flow_matching_action_head.py get_action: pred = action_decoder(model_output)). Its input is
still a noisy chunk (t = 0.75), so _feat_clean gives the policy's representation of the chunk actually executed.
usage (cwd third_party/Isaac-GR00T, like serve_seeded.py):
  python $ROBOCASA_ROOT/runs/scripts/serve_features.py --model_path <checkpoint dir> --port 5791
"""
import serve_seeded  # first: sets CUBLAS_WORKSPACE_CONFIG and the deterministic flags before CUDA initialises
import argparse
import numpy as np
import torch


class FeatureServer(serve_seeded.SeededServer):
    def __init__(self, model, host="*", port=5555, api_token=None):
        super().__init__(model, host=host, port=port, api_token=api_token)
        head = model.model.action_head  # after Gr00tPolicy's init, which may rebuild the head (policy.py:248)
        horizon, outs = head.config.action_horizon, []

        def hook(module, inputs, output):
            outs.append(output[0] if isinstance(output, tuple) else output)
        head.model.register_forward_hook(hook)
        sample = head.get_action

        def get_action_and_clean_pass(backbone_output, action_input):
            # GR00T_N1_5.get_action calls self.action_head.get_action, so this instance attribute is on the call path
            out = sample(backbone_output, action_input)  # processes backbone_output in place (process_backbone_output)
            a, vl, eid = out["action_pred"], backbone_output.backbone_features, action_input.embodiment_id
            ts = torch.full((a.shape[0],), head.num_timestep_buckets - 1, device=a.device)
            af = head.action_encoder(a, ts, eid)  # the loop body of get_action, with the finished chunk
            if head.config.add_pos_embed:
                af = af + head.position_embedding(torch.arange(af.shape[1], dtype=torch.long, device=a.device)).unsqueeze(0)
            ft = head.future_tokens.weight.unsqueeze(0).expand(vl.shape[0], -1, -1)
            head.model(hidden_states=torch.cat((head.state_encoder(action_input.state, eid), ft, af), dim=1),
                       encoder_hidden_states=vl, timestep=ts)
            return out
        head.get_action = get_action_and_clean_pass
        seeded = self._endpoints["get_action"].handler

        def get_action(obs):
            outs.clear()
            out = seeded(obs)
            n = head.num_inference_timesteps
            if len(outs) != n + 1:
                raise RuntimeError(f"DiT ran {len(outs)} times, expected {n} denoising steps + 1 clean pass")
            if outs[-1].shape[0] != 1:
                raise RuntimeError(f"batch {outs[-1].shape[0]}: the features assume one observation per request")
            act = [o[0, -horizon:].float().cpu().numpy() for o in outs]
            out["_feat"], out["_feat_clean"] = act[n - 1], act[n]
            out["_feat_steps"] = np.stack([x.mean(0) for x in act[:n]])
            out["_n_dit"] = len(outs)
            return out
        self.register_endpoint("get_action", get_action)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--data_config", default="panda_omron")  # run_eval.py defaults
    ap.add_argument("--embodiment_tag", default="new_embodiment")
    a = ap.parse_args()
    print(f"serve_features: torch {torch.__version__}, deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}",
          flush=True)
    serve_seeded.run_eval.RobotInferenceServer = FeatureServer  # run_server() looks the class up in its module globals
    serve_seeded.run_eval.run_server(data_config=a.data_config, model_path=a.model_path, embodiment_tag=a.embodiment_tag,
                                     port=a.port)
