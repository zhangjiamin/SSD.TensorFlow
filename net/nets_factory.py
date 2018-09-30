from net import ssd_net
from net import ssd_net_a
from net import rfb_ssd_net
from net import fpn_ssd_net

network_map = {
    'ssd_net':ssd_net,
    'ssd_net_a': ssd_net_a,
    'rfb_ssd_net': rfb_ssd_net,
    'fpn_ssd_net': fpn_ssd_net,
}

def get_network(name):
    return network_map[name]

