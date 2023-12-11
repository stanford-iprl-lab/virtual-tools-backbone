from typing import Tuple, Annotated, Dict
from ..interfaces import VTInterface
from ..world import VTWorld
from abc import ABC, abstractmethod
import copy, random

__all__ = ['VTGenerator', 'StaticGenerator']


"""
Keyword arguments:
argument -- description
Return: return_description
"""

class VTGenerator(ABC):
    
    _opts = {}
    
    def __init__(self, options: Dict):
        # Initialize arguments
        self.set_options(**options)
    
    # Method to be overwritten that creates the world (specific to each interface)
    @abstractmethod
    def propose_world(self) -> VTInterface:
        raise NotImplementedError("Must overwrite propose_world method in inheritor")

    # Method to overwrite that defines any specific action placements to check
    # Note that this is only called if the options min_spec_place or max_spec_place are called
    # Takes an argument of a VTWorld in case you want the specific action to depend
    #  on how the VTWorld is laid out
    def random_specific_action(self, interface: VTInterface) -> Dict:
        raise NotImplementedError("Must overwrite random_specific_action method in inheritor if specific constraints are set")
    
    def generate_world(self,
                       maxiters: int = 10000,
                       seed: int = None,
                       verbose: bool = False) -> VTInterface:
        if seed:
            random.seed(seed)
        # Loop until a world is generated or it times out
        loopn = 0
        while loopn < maxiters:
            if verbose:
                print('\n-----------\nStarting loop ', loopn)
            interface = self.propose_world()
            if verbose:
                print('Generated world')
            if self.check_placements(interface, verbose):
                if verbose:
                    print('Passed checks - returning')
                return interface
            if verbose:
                'Failed checks - restarting'
            loopn += 1
        
        print('Failed to create world within maximum number of iterations')
        return None
            
    
    # Returns a boolean specifying if the randomly generated actions are within the option bounds    
    def check_placements(self, interface: VTInterface, verbose: bool = False) -> bool:
        minap = self._opts['min_any_place']
        maxap = self._opts['max_any_place']
        if minap > 0.0 and maxap < 1.0:
            anyprop = self._check_any_placement(interface)
            if verbose:
                print('Proportion of any random successes:', anyprop)
            if anyprop < minap:
                if verbose:
                    print('Failed: any random placement too low')
                return False
            if anyprop > maxap:
                if verbose:
                    print('Failed: any random placement too high')
                return False
        
        minsp = self._opts['min_spec_place']
        maxsp = self._opts['max_spec_place']
        if minsp > 0.0 and maxsp < 1.0:
            specprop = self._check_specific_placement(interface)
            if verbose:
                print('Proportion of specific random successes:', specprop)
            if specprop < minsp:
                if verbose:
                    print('Failed: specific random placement too low')
                return False
            if specprop > maxsp:
                if verbose:
                    print('Failed: specific random placement too high')
                return False
            
        if verbose:
            print("Passed all random action checks")
        return True
    
    # Methods that should be overwritten for specific interfaces
    @abstractmethod    
    def generate_random_placement(self, interface: VTInterface) -> Dict:
        raise NotImplementedError('This should be implemented by an interface generator')
    
    # Method that returns the proportion of random actions that are successful
    def _check_any_placement(self, interface: VTInterface) -> float:
        numsuc = 0
        for _ in range(self._opts['nsims']):
            act = self.generate_random_placement(interface)
            r = None
            # Ignore failed placements
            while r is None:
                act = self.generate_random_placement(interface)
                r = interface.run_placement(act)[0]
            if r:
                numsuc += 1
        return numsuc / self._opts['nsims']
    
    # Method returning proportion of actions defined in random_specific_action that are successful
    def _check_specific_placement(self, interface: VTInterface) -> float:
        numsuc = 0
        for _ in range(self._opts['nsims']):
            act = self.random_specific_action(interface)
            r = None
            while r is None:
                act = self.random_specific_action(interface)
                r = interface.run_placement(act)[0]
            if r:
                numsuc += 1
        return numsuc / self._opts['nsims']

    # Function to set defaults options (see above)
    def set_options(self, **kwargs):
        def _set_opt(onm, default):
            self._opts[onm] = kwargs.get(onm, self._opts.get(onm, default))
            
        _set_opt('min_any_place', 0.0)
        _set_opt('max_any_place', 1.0)
        _set_opt('min_spec_place', 0.0)
        _set_opt('max_spec_place', 1.0)
        _set_opt('nsims', 100)
        
    @property
    def options(self):
        return copy.deepcopy(self._opts)


# A stub abstract class for generating worlds without randomness
# Note that this is typically only used 
class StaticGenerator(VTGenerator):
    
    def generate_random_placement(self, interface: VTInterface) -> Dict:
        raise NotImplementedError('This function should never be called!')
        
    def set_options(self, **kwargs):
        badkws = ['min_any_place', 'max_any_place', 'min_spec_place', 'max_spec_place']
        if any([kw in kwargs.keys() for kw in badkws]):
            raise Exception('Cannot set acceptability ranges for StaticGenerators')
        return super().set_options(**kwargs)