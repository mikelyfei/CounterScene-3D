"""Calibrated perpendicular front/side/top cameras for native reconstruction."""
import math
from .spatial_suite.raycast import camera


def three_view_cameras(room):
    if not isinstance(room,(list,tuple)) or len(room)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in room):
        raise ValueError('Three finite positive room dimensions required')
    centre=[v/2 for v in room];radius=1.4*math.sqrt(sum(v*v for v in room))
    eyes=dict(front=[centre[0],centre[1]-radius,centre[2]],side=[centre[0]+radius,centre[1],centre[2]],top=[centre[0],centre[1],centre[2]+radius])
    # These are perpendicular calibrated perspective views, not an unsupported
    # orthographic renderer. Top uses a non-collinear camera-up vector.
    result={name:camera(eye,centre,fov_deg=50,width=640,height=480) for name,eye in eyes.items()}
    result['top']['up']=[0,1,0]
    return result
