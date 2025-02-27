import {DQNAgent} from "./dqn-agent";
import {Net} from "../net";

describe("DQNAgent", () => {
  describe("toJSON", () => {
    it("should be correct shape", () => {
      const agent = new DQNAgent({
        gamma: 1,
        epsilon: 2,
        alpha: 3,
        experienceAddEvery: 4,
        experienceSize: 5,
        learningStepsPerIteration: 6,
        tdErrorClamp: 7,
        hiddenLayers: [8],
        inputSize: 9,
        outputSize: 10,
        activation: "sigmoid",
      });
      const json = agent.toJSON();
      expect(json).toEqual({
        gamma: 1,
        epsilon: 2,
        alpha: 3,
        experienceAddEvery: 4,
        experienceSize: 5,
        learningStepsPerIteration: 6,
        tdErrorClamp: 7,
        hiddenLayers: [8],
        inputSize: 9,
        outputSize: 10,
        activation: "sigmoid",
        net: agent.net.toJSON(),
      });
    });
  });
  describe("constructor", () => {
    it("can serialize from json", () => {
      const agent = new DQNAgent({
        gamma: 1,
        epsilon: 2,
        alpha: 3,
        experienceAddEvery: 4,
        experienceSize: 5,
        learningStepsPerIteration: 6,
        tdErrorClamp: 7,
        hiddenLayers: [8],
        inputSize: 9,
        outputSize: 10,
        activation: "sigmoid",
      });
      expect(new DQNAgent(agent.toJSON()).toJSON()).toMatchObject(agent.toJSON());
    });
  });
});
