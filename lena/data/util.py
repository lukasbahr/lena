import lena.data.system as System

def get_system(params):
    if params['system'] == 'van_der_pohl':
        system = System.ClassicVanDerPohl()
    elif params['system'] == 'rev_duffing':
        system = System.ClassicRevDuffing()
    
    system.set_controller(params['init_controller'])

    return system