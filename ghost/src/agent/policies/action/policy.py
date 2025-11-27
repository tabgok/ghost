from abc import abstractmethod

class ActionPolicy:
    @abstractmethod
    def act(self, action_space, observation):
        pass
