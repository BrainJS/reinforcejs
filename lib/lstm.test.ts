import {LSTM} from "./lstm";
import {Mat} from "./mat";
import {Graph} from "./graph";

jest.mock("./rand-mat", () => {
  const { Mat } = require("./mat");
  return {
    RandMat: Mat,
  };
});

describe("LSTM", () => {
  describe("constructor", () => {
    const lstm = new LSTM(1,[2],3);
    it("saves inputs on properties", () => {
      expect(lstm.inputSize).toEqual(1);
      expect(lstm.hiddenLayers).toEqual([2]);
      expect(lstm.outputSize).toEqual(3);
    });
    it("sets up model", () => {
      expect(lstm.model).toEqual({
        layers: [{
          Wix: new Mat(2, 1),
          Wih: new Mat(2, 2),
          bi: new Mat(2, 1),
          Wfx: new Mat(2, 1),
          Wfh: new Mat(2, 2),
          bf: new Mat(2, 1),
          Wox: new Mat(2, 1),
          Woh: new Mat(2, 2),
          bo: new Mat(2, 1),
          Wcx: new Mat(2, 1),
          Wch: new Mat(2, 2),
          bc: new Mat(2, 1),
        }],
        Whd: new Mat(3, 0),
        bd: new Mat(3, 1),
      });
    });
  });
  describe("forward", () => {
    describe("when prev is null", () => {
      it("returns a hiddenCellOutput", () => {
        const lstm = new LSTM(2, [2], 2);
        const graph = new Graph();
        const out = lstm.forward(graph, [2], new Mat(2, 1), null);
        expect(out).toEqual({
          hidden: [new Mat(2, 1)],
          cell: [new Mat(2, 1)],
          output: new Mat(2, 1),
        });
      });
    });
    describe("when prev is ILSTMCell", () => {
      it("returns a hiddenCellOutput", () => {
        const lstm = new LSTM(2, [2], 2);
        const graph = new Graph();
        const firstOut = lstm.forward(graph, [2], new Mat(2, 1), null);
        const secondOut = lstm.forward(graph, [2], new Mat(2, 1), firstOut);
        expect(secondOut).toEqual({
          hidden: [new Mat(2, 1)],
          cell: [new Mat(2, 1)],
          output: new Mat(2, 1),
        });
      });
    });
  });
  describe("update", () => {
    let lstm = new LSTM(2, [2], 2);
    beforeEach(() => {
      jest.spyOn(lstm.model.layers[0].Wix, "update");
      jest.spyOn(lstm.model.layers[0].Wih, "update");
      jest.spyOn(lstm.model.layers[0].bi, "update");
      jest.spyOn(lstm.model.layers[0].Wfx, "update");
      jest.spyOn(lstm.model.layers[0].Wfh, "update");
      jest.spyOn(lstm.model.layers[0].bf, "update");
      jest.spyOn(lstm.model.layers[0].Wox, "update");
      jest.spyOn(lstm.model.layers[0].Woh, "update");
      jest.spyOn(lstm.model.layers[0].bo, "update");
      jest.spyOn(lstm.model.layers[0].Wcx, "update");
      jest.spyOn(lstm.model.layers[0].Wch, "update");
      jest.spyOn(lstm.model.layers[0].bc, "update");
      jest.spyOn(lstm.model.Whd, "update");
      jest.spyOn(lstm.model.bd, "update");
    });
    it("calls update on all layers", () => {
      const alpha = 0.5;
      lstm.update(alpha);
      expect(lstm.model.layers[0].Wix.update).toHaveBeenCalledWith(alpha);
      expect(lstm.model.layers[0].Wih.update).toHaveBeenCalledWith(alpha);
      expect(lstm.model.layers[0].bi.update).toHaveBeenCalledWith(alpha);
      expect(lstm.model.layers[0].Wfx.update).toHaveBeenCalledWith(alpha);
      expect(lstm.model.layers[0].Wfh.update).toHaveBeenCalledWith(alpha);
      expect(lstm.model.layers[0].bf.update).toHaveBeenCalledWith(alpha);
      expect(lstm.model.layers[0].Wox.update).toHaveBeenCalledWith(alpha);
      expect(lstm.model.layers[0].Woh.update).toHaveBeenCalledWith(alpha);
      expect(lstm.model.layers[0].bo.update).toHaveBeenCalledWith(alpha);
      expect(lstm.model.layers[0].Wcx.update).toHaveBeenCalledWith(alpha);
      expect(lstm.model.layers[0].Wch.update).toHaveBeenCalledWith(alpha);
      expect(lstm.model.layers[0].bc.update).toHaveBeenCalledWith(alpha);
      expect(lstm.model.Whd.update).toHaveBeenCalledWith(alpha);
      expect(lstm.model.bd.update).toHaveBeenCalledWith(alpha);
    });
  });
});
