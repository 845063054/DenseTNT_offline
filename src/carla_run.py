from ast import If
import os
from re import T
import sys
from tkinter import N
import carla
import math
import csv
import logging
from yaml_utils import load_yaml
from numpy import random
import time
from carla_submap_wrapper import get_all_lane_info

class CarlaSyncModeWithTraffic(object):
    """
    Carla client manager with traffic
    """
    def __init__(self):
        config_yaml = os.path.join(os.path.dirname(os.path.realpath(__file__)),'carla_config.yaml')
        if not os.path.isfile(config_yaml):
            sys.exit("carla_config.yaml not found!")
        self.config_params = load_yaml(config_yaml)
        self.world_params = self.config_params['world']
        self.save_parameters = self.config_params['save_parameters']
        self.trajectory_path = self.save_parameters['trajectory_path'] + self.world_params["map_name"]
        
        self.trajectory_number = 0
        self.data_number = 0
        self.tick_number = 0
        
        self.csvfile = None
        self.writer = None
        self.save_trajectory_item()
        self.init_time = time.time()
        
        # set client and world config
        self.client = carla.Client('localhost', self.world_params["client_port"])
        self.client.set_timeout(self.world_params["timeout"])
        self.client.load_world(self.world_params["map_name"])
        self.world = self.client.get_world()
        self.origin_settings = self.world.get_settings()
        # set weather
        self._set_weather()
        self.world.set_weather(self.weather)
        # set hero vehicle and others
        self.vehicles_list = []
        self.hero_actor = None
        self._select_hero_actor()
        self.hero_actor.set_autopilot(True)
        # set traffic 
        self._setup_traffic_manager()
        self.map = self.world.get_map()
        get_all_lane_info(self.trajectory_path,self.map)

    def tick(self):
        self.world.tick()
        self.tick_number += 1
        if self.hero_actor is not None:
            self.hero_transform = self.hero_actor.get_transform()
        # set spectator
        spectator_location = self.hero_transform.location + carla.Location(z = 300)
        spectator_rotation = carla.Rotation(pitch = -90)
        spectator_transform = carla.Transform(spectator_location, spectator_rotation)
        spectator = self.world.get_spectator()
        spectator.set_transform(spectator_transform)
        if self.tick_number >= self.save_parameters['save_number']:
            self.save_trajectory_data()
            self.tick_number = 0
        
    def save_trajectory_item(self):
        # set the path of trajectory 
        if not os.path.isdir(self.trajectory_path):
            os.makedirs(self.trajectory_path)
        trajectory_number_str = str(self.trajectory_number).zfill(4)
        trajectory_path_csv = self.trajectory_path + '/' + trajectory_number_str +'.csv'
        self.csvfile = open(trajectory_path_csv,'w',newline='')
        self.writer = csv.writer(self.csvfile)
        item = ['TIMESTAMP','TRACK_ID','OBJECT_TYPE','X','Y','CITY_NAME']
        self.writer.writerow(item)
        self.init_time = time.time()

    def save_trajectory_data(self):
        max_trajectory_size = self.save_parameters['max_trajectory_size']
        if self.data_number >= max_trajectory_size:
            self.data_number = 0
            self.trajectory_number += 1
            self.csvfile.close()
            self.save_trajectory_item() 
        self.data_number += 1
        for i in range(len(self.vehicles_list)):
            actor_i = self.world.get_actor(self.vehicles_list[i])
            pos = actor_i.get_location() 
            # print( actor_i.attributes['role_name'])
            if actor_i.attributes['role_name'] == 'hero':
                track_id = str(0).zfill(6)
                time_current = time.time() - self.init_time
                value = [time_current,track_id,'AGENT',pos.x,pos.y,self.world_params["map_name"]]
                self.writer.writerow(value)
            else:
                track_id = actor_i.id
                # add log
                # print( actor_i.id)
                track_id = str(track_id).zfill(6)
                time_current = time.time() - self.init_time
                value = [time_current,track_id,'OTHERS',pos.x,pos.y,self.world_params["map_name"]]
                self.writer.writerow(value)
        

    def _setup_traffic_manager(self):
        world, client = self.world, self.client
        traffic_manager_params = self.config_params['traffic_manager']
        traffic_manager = client.get_trafficmanager() # Port to communicate with TM (default: 8000)
        traffic_manager.set_global_distance_to_leading_vehicle(traffic_manager_params["global_distance_to_leading_vehicle"])
        traffic_manager.ignore_signs_percentage(self.hero_actor, traffic_manager_params['ignore_signs_percentage'])
        for actor in self.world.get_actors():
            if actor.type_id == 'traffic.traffic_light':
                actor.freeze(True)
                actor.set_state(carla.TrafficLightState.Green)

        if traffic_manager_params["respawn_dormant_vehicles"]:
            traffic_manager.set_respawn_dormant_vehicles(traffic_manager_params["respawn_dormant_vehicles"])
        if traffic_manager_params["hybrid_physics_mode"]:
            traffic_manager.set_hybrid_physics_mode(traffic_manager_params["hybrid_physics_mode"])
            traffic_manager.set_hybrid_physics_radius(traffic_manager_params["hybrid_physics_radius"])
        if traffic_manager_params["random_device_seed"] is not None:
            traffic_manager.set_random_device_seed(traffic_manager_params["random_device_seed"])
        
        settings = self.world.get_settings()
        traffic_manager.set_synchronous_mode(traffic_manager_params['sync_mode'])
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = traffic_manager_params['sync_mode']
            settings.fixed_delta_seconds = traffic_manager_params['fixed_delta_seconds']
            settings.substeping = traffic_manager_params['substeping']
            settings.max_substeps = traffic_manager_params['max_substeps']
            settings.max_substep_delta_time = traffic_manager_params['max_substep_delta_time']
        else:
            synchronous_master = False
        world.apply_settings(settings)
        blueprints = self._get_actor_blueprints(traffic_manager_params['filterv'], traffic_manager_params['generationv'])
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        number_of_vehicles = traffic_manager_params['number_of_vehicles']
        if number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, self.number_of_vehicles, number_of_spawn_points)
            number_of_vehicles = number_of_spawn_points

        # cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # Spawn vehicles
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        print('spawned %d vehicles, press Ctrl+C to exit.' % (len(self.vehicles_list)))
        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(traffic_manager_params['global_percentage_speed_difference'])

    def _get_actor_blueprints(self, filter, generation):
        bps = self.world.get_blueprint_library().filter(filter)
        if generation.lower() == "all": return bps
        # If the filter returns only one bp, we assume that this one needed and therefore, we ignore the generation
        if len(bps) == 1: return bps
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("Actor Generation is not valid. No actor will be spawned.")
            return []

    def _select_hero_actor(self):
        hero_vehicles = [actor for actor in self.world.get_actors(
        ) if 'vehicle' in actor.type_id and actor.attributes['role_name'] == 'hero']
        if len(hero_vehicles) > 0:
            self.hero_actor = random.choice(hero_vehicles)
            self.hero_transform = self.hero_actor.get_transform()
        else:
            # Get a random blueprint.
            blueprint = random.choice(self.world.get_blueprint_library().filter(self.config_params['traffic_manager']['filterv']))
            blueprint.set_attribute('role_name', 'hero')
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            # Spawn the player.
            while self.hero_actor is None:
                spawn_points = self.world.get_map().get_spawn_points()
                spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
                self.hero_actor = self.world.try_spawn_actor(blueprint, spawn_point)
            self.hero_transform = self.hero_actor.get_transform()
            # Save it in order to destroy it when closing program
            self.spawned_hero = self.hero_actor
            self.vehicles_list.insert(0, self.hero_actor.id)

    def destroy_vechicles(self):
        self.world.apply_settings(self.origin_settings)
        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        # if self.spawned_hero is not None:
        #     self.spawned_hero.destroy()
        time.sleep(0.25)
        self.csvfile.close()

    def _set_weather(self):
        """
        Set CARLA weather params.
        """
        weather_settings = self.world_params["weather"]
        self.weather = carla.WeatherParameters(
        sun_altitude_angle = weather_settings['sun_altitude_angle'],
        cloudiness = weather_settings['cloudiness'],
        precipitation = weather_settings['precipitation'],
        precipitation_deposits=weather_settings['precipitation_deposits'],
        wind_intensity = weather_settings['wind_intensity'],
        fog_density = weather_settings['fog_density'],
        fog_distance = weather_settings['fog_distance'],
        fog_falloff = weather_settings['fog_falloff'],
        wetness = weather_settings['wetness'])

if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    carla_client = CarlaSyncModeWithTraffic()
    try:
        while True:
            carla_client.tick()
    finally:
        carla_client.destroy_vechicles()


