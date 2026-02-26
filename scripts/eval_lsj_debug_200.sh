#!/bin/bash

bash ./scripts/launch_vlm_servers.sh

python -um vlfm.run \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval.video_option='["disk"]' \
  habitat_baselines.test_episode_count=200
