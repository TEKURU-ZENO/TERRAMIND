import numpy as np
import pytest

from terramind.sim.env import BatchedFloodResponse, SimConfig, border_cells, make_gym_env
from terramind.sim.policies import Assignment, Greedy, Lookahead, RandomWalk, rollout
from terramind.sim.scenarios import cells

G = 8


def maps(n=3, seed=0):
    rng = np.random.default_rng(seed)
    truth = np.zeros((n, G, G), np.float32)
    truth[:, 2:5, 3:7] = rng.random((n, 3, 4))
    pred = np.clip(truth + rng.normal(0, 0.1, truth.shape), 0, 1).astype(np.float32)
    unc = np.full_like(truth, 0.05)
    return truth, pred, unc


def test_cells_average_blocks():
    a = np.zeros((8, 8), np.float32)
    a[:4, :4] = 1.0
    assert cells(a, 2).tolist() == [[1.0, 0.0], [0.0, 0.0]]


def test_border_cells_are_unique_and_on_the_edge():
    b = border_cells(G)
    assert len(b) == len({tuple(x) for x in b}) == 4 * G - 4
    assert (((b == 0) | (b == G - 1)).any(1)).all()


def test_reward_is_normalised_and_cells_pay_once():
    truth, pred, unc = maps()
    env = BatchedFloodResponse(truth, pred, unc, SimConfig(teams=2, horizon=200))
    env.reset([0, 1, 2], [0, 5, 9])
    total = np.zeros(3)
    rng = np.random.default_rng(1)
    while not env.done:
        total += env.step(rng.integers(5, size=(3, 2)))
    m = env.metrics()
    # `served` also counts the depot cell, which is served at reset without reward.
    assert (m["served"] >= total - 1e-5).all() and (m["served"] <= 1 + 1e-5).all()
    assert (m["auc"] <= m["served"] + 1e-6).all()


def test_off_map_moves_keep_the_team_in_place_and_truth_is_revealed_nearby():
    truth, pred, unc = maps()
    env = BatchedFloodResponse(truth, pred, unc, SimConfig(teams=1, horizon=4))
    env.reset([0], [0])  # top-left corner
    assert not env.action_mask()[0, 0, 1] and env.action_mask()[0, 0, 2]
    env.step(np.array([[1]]))  # north, off the map
    assert env.pos[0, 0].tolist() == [0, 0]
    assert env.seen[0, :2, :2].all() and not env.seen[0, 3, 3]
    assert env.belief()[0, 1, 1] == truth[0, 1, 1] and env.belief()[0, 3, 3] == pred[0, 3, 3]


def test_observation_is_centred_on_the_agent():
    truth, pred, unc = maps()
    env = BatchedFloodResponse(truth, pred, unc, SimConfig(teams=2, horizon=4))
    env.reset([0, 1], [3, 7])
    actor, critic = env.observe()
    assert actor.shape == (2, 2, 7, 2 * G - 1, 2 * G - 1) and critic.shape[2] == 8
    centre = G - 1
    assert (actor[:, :, 5, centre, centre] == 1).all()  # on-map
    assert (actor[:, :, 2, centre, centre] == 1).all()  # own cell already served
    assert (actor[:, :, 4, centre, centre] == 1).all()  # the other team starts here too
    r, c = env.pos[0, 0]
    assert np.allclose(
        critic[0, 0, 7, centre - r : centre - r + G, centre - c : centre - c + G], truth[0]
    )


def test_informed_policies_beat_random_and_oracle_beats_belief():
    truth, pred, unc = maps(n=6)
    env = BatchedFloodResponse(truth, pred, unc, SimConfig(teams=2, horizon=12))
    scenario, depot = np.arange(6), np.array([0, 4, 8, 12, 16, 20])
    score = {
        name: rollout(env, p, scenario, depot)["auc"].mean()
        for name, p in {
            "random": RandomWalk(),
            "greedy": Greedy(),
            "assignment": Assignment(),
            "lookahead": Lookahead(depth=3),
            "oracle": Lookahead(depth=3, value="oracle"),
        }.items()
    }
    assert min(score["greedy"], score["assignment"], score["lookahead"]) > score["random"]
    assert score["oracle"] >= score["lookahead"] - 0.02


def test_gymnasium_api():
    pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env

    truth, pred, unc = maps()
    env = make_gym_env(truth, pred, unc, SimConfig(teams=2, horizon=5))
    check_env(env, skip_render_check=True)
    _, info = env.reset(seed=3, options={"scenario": 1, "depot": 2})
    assert info == {"scenario": 1, "depot": 2}
    for _ in range(5):
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
    assert truncated and not terminated and 0 <= info["served"] <= 1


def test_dispatch_only_deploying_serves_and_dry_deployments_cost():
    from terramind.sim.dispatch import DEPLOY, DispatchConfig, DispatchEnv

    truth = np.zeros((1, G, G), np.float32)
    truth[0, 0, 1] = 0.8
    env = DispatchEnv(
        truth, truth, np.zeros_like(truth), DispatchConfig(teams=1, horizon=6, waste_cost=0.5)
    )
    env.reset([0], [0])  # top-left corner, a dry cell
    assert env.metrics()["served"][0] == 0 and not env.seen.any()
    assert env.step(np.array([[4]]))[0] == 0  # moving east onto water serves nothing
    assert env.step(np.array([[DEPLOY]]))[0] == pytest.approx(1.0)
    assert env.seen[0, :2, :3].all() and not env.action_mask()[0, 0, DEPLOY]
    assert env.step(np.array([[DEPLOY]]))[0] == 0  # the cell is already served
    env.step(np.array([[3]]))  # back west to the dry corner
    assert env.step(np.array([[DEPLOY]]))[0] == pytest.approx(-0.5 / 0.8)
    m = env.metrics()
    assert m["wasted"][0] == 1 and m["utility"][0] == pytest.approx(1 - 0.5 / 0.8)


def test_dispatcher_avoids_risky_cells_as_waste_cost_rises_and_oracle_never_wastes():
    from terramind.sim.dispatch import Calibrated, DispatchConfig, DispatchEnv, Dispatcher, rollout

    truth, pred, unc = maps(n=6, seed=3)
    # A band the model wrongly believes is slightly wet.
    pred[:, 6, :] = 0.15
    scenario, depot = np.arange(6), np.arange(6) * 3
    wasted = []
    for cost in (0.1, 2.0):
        cfg = DispatchConfig(teams=2, horizon=40, waste_cost=cost)
        belief = Calibrated().fit(pred, unc, truth, cfg.dry)
        wasted.append(
            rollout(DispatchEnv(truth, pred, unc, cfg), Dispatcher(belief), scenario, depot)[
                "wasted"
            ].mean()
        )
        oracle = rollout(
            DispatchEnv(truth, pred, unc, cfg), Dispatcher(None, oracle=True), scenario, depot
        )
        assert oracle["wasted"].sum() == 0 and (oracle["utility"] > 0).all()
    assert wasted[1] <= wasted[0]


def test_dispatch_observation_gives_each_team_its_own_id():
    from terramind.sim.dispatch import DispatchEnv

    truth, pred, unc = maps()
    env = DispatchEnv(truth, pred, unc)
    env.reset([0, 1], [3, 7])
    actor, critic = env.observe()
    assert actor.shape[2] == env.actor_channels and critic.shape[2] == env.critic_channels
    assert [float(actor[0, k, 7, 0, 0]) for k in range(3)] == [0.0, 0.5, 1.0]
    assert env.action_mask().shape == (2, 3, env.n_actions)
