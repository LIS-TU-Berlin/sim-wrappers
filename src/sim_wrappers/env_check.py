import argparse
import numpy as np
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt 

from sim_wrappers.mujoco_gym import MujocoGymGoal

def feature_map(qpos, qvel, scale_obj=2., scale_q=1.):

    # object position - last 3 positions of qpos
    z_obj = qpos[-3:]

    # robot angles (=finger position) - everything before last 3 positions
    z_q = qpos[:-3]

    # scaled concatenation
    return np.concat((scale_obj*z_obj, scale_q*z_q))
        
def main(args):

    env = MujocoGymGoal(
        config_path=args.config_path,
        xml_path=args.xml_path,
        scene_path=args.scene_path,
        sparse_r_thr=args.sparse_r_thr,
        feature_map=feature_map,
        engine='mujoco',
        start_id=-1,
        goal_id=-1,
    )

    # SB3 check_env
    print("Running check_env...")
    check_env(env, warn=True)
    print("check_env passed!")

    # Single rollout
    obs, _ = env.reset()
    done = False
    step_count = 0
    print("Running a single rollout...")
    while not done and step_count < 10:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Check observation dict keys
        assert all(k in obs for k in ["observation", "achieved_goal", "desired_goal"])
        # Check reward type
        assert isinstance(reward, float), f"Reward is not float! Got {type(reward)}"
        step_count += 1

    print(f"Rollout finished after {step_count} steps. Rewards and obs OK.")

    # Rendering
    if hasattr(env, "render_mode"):
        env.render_mode = "rgb_array"
        img = env.render()
        assert isinstance(img, np.ndarray), "Render did not return an image"
        print("Render OK. Image shape:", img.shape)
        plt.imshow(img)

    env.close()
    print("Environment check completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check MujocoGymGoal for SB3/Gymnasium compatibility")
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--xml_path", type=str, required=True)
    parser.add_argument("--scene_path", type=str, required=True)
    parser.add_argument("--sparse_r_thr", type=float, default=10.0)
    
    args = parser.parse_args()
    main(args)
