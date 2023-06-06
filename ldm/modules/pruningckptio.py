from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO

from typing import Any, Dict, Optional

#from lightning_lite.utilities.types import _PATH
from pytorch_lightning.utilities.types import _PATH
from ldm.pruner import prune_checkpoint

class PruningCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(self,
                        checkpoint: Dict[str, Any],
                        path: _PATH,
                        storage_options: Optional[Any] = None) -> None:
        if 'last.ckpt' in path:
            print('Saving unpruned "last.ckpt"...')
            TorchCheckpointIO.save_checkpoint(self, checkpoint, path, storage_options)
        else:
            pruned_checkpoint = prune_checkpoint(checkpoint)
            TorchCheckpointIO.save_checkpoint(self, pruned_checkpoint, path, storage_options)
