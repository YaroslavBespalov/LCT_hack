from typing import List, Callable, Union, Tuple, Optional

import torch
from torch import nn, Tensor

import copy


class GoldTuner:
    def __init__(self, coefs: List[float], device, rule_eps: float, radius: float, active=True):
        self.coefs = torch.tensor(coefs, dtype=torch.float32, device=device)
        self.y: dict = {"y1": None, "y2": None}
        self.x1: Optional[Tensor] = None
        self.x2: Optional[Tensor] = None
        self.a: Optional[Tensor] = None
        self.b: Optional[Tensor] = None
        self.proector = lambda x: x.clamp_(0, 1000)
        self.radius = radius
        self.queue = []
        self.active = active
        if active:
            self.direction()
        else:
            self.queue = [("0", self.coefs)]
        self.rule_eps = rule_eps
        self.prev_opt_y = None
        self.directions_tested = 0
        self.direction_score = 100
        self.best_ab: Optional[Tuple[Tensor, Tensor]] = None
        self.repeat_coefs = False

    def update_coords(self):
        self.x1 = self.a + 0.382 * (self.b - self.a)
        self.x2 = self.b - 0.382 * (self.b - self.a)
        self.queue.append(("1", self.x1))
        self.queue.append(("2", self.x2))

    def direction(self):
        assert len(self.queue) == 0
        rand_direction = torch.randn_like(self.coefs)
        rand_direction_norm = torch.sqrt((rand_direction ** 2).sum())
        rand_direction = rand_direction / rand_direction_norm
        self.a = self.coefs
        self.b = self.proector(self.a + self.radius * rand_direction)
        print(f"Choose direction from {self.a} to {self.b}")
        self.update_coords()

    def get_coef(self):
        c = self.coefs if self.repeat_coefs else self.queue[0][1]
        # print("current test with coefs:", c)
        return c

    def update(self, y: float):

        if not self.active:
            return None

        # if self.repeat_coefs:
        #     print("repeat with stable coefs")
        #     self.repeat_coefs = False
        #     self.direction()
        #     return None

        self.y["y" + self.queue[0][0]] = y
        self.queue.pop(0)
        if len(self.queue) == 0:

            if self.prev_opt_y and self.y["y1"] > self.prev_opt_y and self.y["y2"] > self.prev_opt_y and self.directions_tested < 5:
                print("Take new direction")
                self.directions_tested += 1

                score = min(self.y["y1"], self.y["y2"])
                if self.direction_score > score:
                    self.best_ab = (self.a, self.b)
                    self.direction_score = score

                self.direction()
                # self.repeat_coefs = True
                # return None

            elif self.directions_tested >= 5:
                print(f"Take best from 5 direction {self.a} to {self.b}")
                self.a = self.best_ab[0]
                self.b = self.best_ab[1]
                self.update_coords()
                self.prev_opt_y = 100
                self.directions_tested = 0
                self.direction_score = 100
                self.best_ab = None
                return None

            self.directions_tested = 0
            self.direction_score = 100
            self.best_ab = None

            if self.stop_rule():
                print("opt val:", self.prev_opt_y)
                self.direction()
                return None

            self.optimize()


    def optimize(self):
        print(self.y)
        if self.y["y1"] >= self.y["y2"]:
            self.a = self.x1
            self.x1 = self.x2
            self.x2 = self.b - 0.382 * (self.b - self.a)
            self.queue.append(("2", self.x2))
            self.y["y1"] = self.y["y2"]
            print("new a: ", self.a)
        else:
            self.b = self.x2
            self.x2 = self.x1
            self.x1 = self.a + 0.382 * (self.b - self.a)
            self.queue.append(("1", self.x1))
            self.y["y2"] = self.y["y1"]
            print("new b: ", self.b)

    def stop_rule(self):
        if (self.b - self.a).abs().max() < self.rule_eps:
            self.coefs = (self.a + self.b) / 2
            self.prev_opt_y = (self.y["y1"] + self.y["y2"]) / 2
            self.x1 = None
            self.x2 = None
            self.queue = []
            print("coefs: ", self.coefs)
            return True
        return False

    # def sum_losses(self, losses: List[Loss]) -> Loss:
    #     res = Loss.ZERO()
    #     coef = self.get_coef()
    #     for i, l in enumerate(losses):
    #         res = res + l * coef[i].detach()
    #
    #     return res






