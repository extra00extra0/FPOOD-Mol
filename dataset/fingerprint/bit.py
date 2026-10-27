# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 15:09:26 2022

@author: Wanxiang.shen

Basic bit
"""

from rdkit.Chem.Draw import rdMolDraw2D

class Bit(object):
    
    def __init__(self, onebitdict={'idx': None,
                                   'name': None,
                                   'on': None,
                                   'patt': None,
                                   'smarts':None,
                                   'info': None,
                                   'descr': None,
                                   'atomInUse': None,
                                   'bondInUse': None
                                   }):
        self.__dict__.update(onebitdict)

    def show(self, mol,
              width=300,
              height=300,
              highlightAtomColors=None,
              highlightBondColors=None,
              highlightAtomRadii=None, **kwargs):
        '''
        highlight the bit on a given mol, refer to the rdMolDraw2D.PrepareAndDrawMolecule method
        '''
        atomsToUse = self.atomInUse
        bondsToUse = self.bondInUse
        # or MolDraw2DCairo to get PNGs
        d = rdMolDraw2D.MolDraw2DSVG(width, height)
        rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=atomsToUse,
                                           highlightBonds=bondsToUse,
                                           highlightBondColors=highlightBondColors,
                                           highlightAtomColors=highlightAtomColors,
                                           highlightAtomRadii=highlightAtomRadii,
                                           **kwargs)
        d.FinishDrawing()
        svg = d.GetDrawingText()
        try:
            from IPython.display import SVG
            return SVG(svg)
        except ImportError:
            return svg
    
    
class FP(object):

    def bits2df(self, bits):
        import pandas as pd
        infos = []
        for bit in bits:
            onebitdict={'idx': bit.idx,
                       'name': bit.name,
                       'on': bit.on,
                       'patt': bit.patt,
                       'smarts':bit.smarts,
                       'info': bit.info,
                       'descr': bit.descr,
                       'atomInUse': bit.atomInUse,
                       'bondInUse': bit.bondInUse}    
            infos.append(onebitdict)
        dfbit = pd.DataFrame(infos)
        return dfbit
