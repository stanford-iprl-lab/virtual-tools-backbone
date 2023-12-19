from typing import Tuple, Annotated, Dict
from ..interfaces import VTInterface
from ..world import VTWorld
from abc import ABC, abstractmethod
import copy, random
import warnings

__all__ = ['VTGenerator', 'StaticGenerator']

class VTGenerator(ABC):
    """An abstract class used as the basis for level generators
    
    See https://docs.google.com/document/d/1tmdCncGLO2KA2Y7aYs5WW07VNhIiKHvu8CAJ1pkZBEY/edit?usp=sharing for more details
    """
    
    _opts = {}
    
    def __init__(self, options: Dict):
        """Initialize a VTGenerator. Though this is abstract, most inheritors won't need to overwrite this
        
        The `options` dictionary can have the following items 

        Args:
            options (Dict): A dictionary with the following 
        """        
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
    
    def random_lure_action(self, interface: VTInterface) -> Dict:
        raise NotImplementedError("Must overwrite random_lure_action method in inheritor if lure constraints are set")
    
    
    def generate_world(self,
                       maxiters: int = 1000,
                       seed: int = None,
                       verbose: bool = False) -> VTInterface:
        """Runs the generator to produce a world within the given constraints

        Args:
            maxiters (int, optional): Total number of failed world creation attempts before giving up and returning None. Defaults to 1000.
            seed (int, optional): The random seed to set for consistent generation. Defaults to None.
            verbose (bool, optional): A flag if detailed output should be written out. Defaults to False.

        Returns:
            VTInterface: A random level proposed by `propose_world` and passing all requested checks. Or None if no level can be found
        """        
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
        """A function to determine whether a given world/interface passes the requested checks

        Args:
            interface (VTInterface): The VTInterface often proposed by `propose_world`
            verbose (bool, optional): A flag if detailed output should be written out. Defaults to False.

        Returns:
            bool: Returns True if all the set checks are passed, False if any are failed
        """        
        minap = self._opts['min_any_place']
        maxap = self._opts['max_any_place']
        if minap > 0.0 and maxap < 1.0:
            anyprop = self._check_placement(interface, 'specific')
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
            specprop = self._check_placement(interface, 'specific')
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
            
        minlp = self._opts['min_lure_place']
        maxlp = self._opts['max_lure_place']
        if minlp > 0.0 and maxlp < 1.0:
            lureprop = self._check_placement(interface, 'lure')
            if verbose:
                print('Proportion of lure random successes:', lureprop)
            if lureprop < minlp:
                if verbose:
                    print('Failed: lure random placement too low')
                return False
            if lureprop > maxlp:
                if verbose:
                    print('Failed: lure random placement too high')
                return False
            
        if verbose:
            print("Passed all random action checks")
        return True
    
    # Methods that should be overwritten for specific interfaces
    @abstractmethod    
    def generate_random_placement(self, interface: VTInterface) -> Dict:
        raise NotImplementedError('This should be implemented by an interface generator')
    
    # Helper method to get the number of successes from a random level
    def _check_placement(self, interface: VTInterface, plc_type: str = "any"):
        if plc_type == 'any':
            fn = self.generate_random_placement
        elif plc_type == 'specific':
            fn = self.random_specific_action
        elif plc_type == 'lure':
            fn = self.random_lure_action
        else:
            raise ValueError("plc_type must be one of 'any', 'specific', or 'lure'")
        numsuc = 0
        for _ in range(self._opts['nsims']):
            act = fn(interface)
            r = None
            # Ignore failed placements
            while r is None:
                act = fn(interface)
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
        _set_opt('min_lure_place', 0.0)
        _set_opt('max_lure_place', 1.0)
        _set_opt('nsims', 100)
        
    @property
    def options(self):
        return copy.deepcopy(self._opts)


# A stub abstract class for generating worlds without randomness
# Note that this is typically only used 
class StaticGenerator(VTGenerator):
    """A class to inheret from in order to generate world with no randomness
    """    
    
    def generate_random_placement(self, interface: VTInterface) -> Dict:
        raise NotImplementedError('This function should never be called!')
        
    def set_options(self, **kwargs):
        badkws = ['min_any_place', 'max_any_place', 'min_spec_place', 'max_spec_place',
                  'min_lure_place', 'max_lure_place']
        if any([kw in kwargs.keys() for kw in badkws]):
            raise Exception('Cannot set acceptability ranges for StaticGenerators')
        return super().set_options(**kwargs)
    
    def generate_world(self,
                       maxiters: int = 1000,
                       seed: int = None,
                       verbose: bool = False) -> VTInterface:
        """Runs the generator to produce a world

        Args:
            maxiters (int, optional): Ignored; only used for inheritance. Defaults to 1000.
            seed (int, optional): The random seed to set for consistent generation. Defaults to None.
            verbose (bool, optional): A flag if detailed output should be written out. Defaults to False.

        Returns:
            VTInterface: A level proposed by `propose_world`
        """        
        if seed:
            random.seed(seed)
            
        if maxiters != 1000 or (verbose is not False):
            warnings.Warn("Note that maxiters and verbose options are unused for static generators")
        return self.propose_world()