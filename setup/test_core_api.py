from omni.isaacsim.core.utils.nucleus import find_nucleus_server
from omni.isaacsim.core.utils.stage import add_reference_to_stage
import carb

def setup_scene(self):
        world = World.instance()
        result, nucleus_server = find_nucleus_server()
        if result is False:
            carb.log_error("Could not find nucleus server with /isaac folder")
        usd_path = nucleus_server + "Scenes/OOJU_Warehouse.usd"
        stand = add_reference_to_stage(usd_path=usd_path, prim_path="/World/Stand")