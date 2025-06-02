######
# This file is used to debug the LLM generated code in Vectorworks
######

import builtins
import ptvsd
import importlib
import tool_agent.vw_tools_extend
importlib.reload(tool_agent.vw_tools_extend)
from tool_agent.vw_tools_extend import *
import tool_agent.python_interpreter
importlib.reload(tool_agent.python_interpreter)
from tool_agent.python_interpreter import evaluate

class DebugServer:
	_instance = None

	@staticmethod
	def getInstance():
		if DebugServer._instance == None:
			DebugServer()
		return DebugServer._instance

	def __init__(self):
		if DebugServer._instance != None:
			raise Exception("This class is a singleton!")
		else:
			DebugServer._instance = self
			svr_addr = str(ptvsd.options.host) + ":" + str(ptvsd.options.port)
			print(" -> Hosting debug server ... (" + svr_addr + ")")
			ptvsd.enable_attach()
			ptvsd.wait_for_attach(0.3)

	def stop(self):
		print(" -> Stopping debug server ...")
		ptvsd.stop()
		DebugServer._instance = None


def test_llm_code():
	# init tools
	create_wall = CreateWallTool()
	set_wall_thickness = SetWallThickness()
	set_wall_elevation = SetWallHeight()
	set_wall_style = SetWallStyle()
	get_wall_elevation = GetWallElevation()
	get_wall_thickness = GetWallThickness()
	add_window_to_wall = AddWindowToWall()
	add_door_to_wall = AddDoorToWall()
	move_obj = Move()
	delete_element = DeleteTool()
	find_selected_element = FindSelect()
	create_polygon = CreatePolygon()
	get_polygon_vertex = GetPolygonVertex()
	get_vertex_count = GetVertNum()
	create_slab = CreateSlab()
	set_slab_height = SetSlabHeight()
	set_slab_style = SetSlabStyle()
	duplicate_obj = DuplicateObj()
	rotate_obj = RotateObj()
	create_pitched_roof = CreateRoof()
	set_pitched_roof_style = SetRoofStyle()
	set_pitched_roof_attributes = SetRoofAttributes()
	create_story_layer = CreateStoryLayer()
	set_active_story_layer = SetStoryLayerActive()
	create_space = CreateSpace()

	# Step 1: Create Story Layers
	# Step 1: Create Story Layers
	ground_floor_uuid = create_story_layer("Ground Floor", 0, 0)
	first_floor_uuid = create_story_layer("First Floor", 3000, 1)
	second_floor_uuid = create_story_layer("Second Floor", 6000, 2)

	# Step 2: Create Perimeter Walls for Each Floor
	# Ground Floor
	wall_A_uuid = create_wall((0, 0), (45000, 0), ground_floor_uuid)
	wall_B_uuid = create_wall((45000, 0), (45000, 15000), ground_floor_uuid)
	wall_C_uuid = create_wall((45000, 15000), (0, 15000), ground_floor_uuid)
	wall_D_uuid = create_wall((0, 15000), (0, 0), ground_floor_uuid)

	# First Floor and Second Floor: Duplicate the walls from the Ground Floor
	first_floor_walls = duplicate_obj(wall_A_uuid, first_floor_uuid, 1) + \
					duplicate_obj(wall_B_uuid, first_floor_uuid, 1) + \
					duplicate_obj(wall_C_uuid, first_floor_uuid, 1) + \
					duplicate_obj(wall_D_uuid, first_floor_uuid, 1)

	second_floor_walls = duplicate_obj(wall_A_uuid, second_floor_uuid, 1) + \
						duplicate_obj(wall_B_uuid, second_floor_uuid, 1) + \
						duplicate_obj(wall_C_uuid, second_floor_uuid, 1) + \
						duplicate_obj(wall_D_uuid, second_floor_uuid, 1)

	# Step 3: Create Central Staircase Walls for Each Floor
	# Ground Floor
	wall_E_uuid = create_wall((20000, 0), (20000, 15000), ground_floor_uuid)
	wall_F_uuid = create_wall((25000, 0), (25000, 15000), ground_floor_uuid)

	# First Floor and Second Floor: Duplicate the walls from the Ground Floor
	first_floor_staircase_walls = duplicate_obj(wall_E_uuid, first_floor_uuid, 1) + \
								duplicate_obj(wall_F_uuid, first_floor_uuid, 1)

	second_floor_staircase_walls = duplicate_obj(wall_E_uuid, second_floor_uuid, 1) + \
								duplicate_obj(wall_F_uuid, second_floor_uuid, 1)

	# Step 4: Create Apartment Dividing Walls for Each Floor
	# Ground Floor
	wall_G_uuid = create_wall((15000, 0), (15000, 15000), ground_floor_uuid)
	wall_H_uuid = create_wall((30000, 0), (30000, 15000), ground_floor_uuid)

	# First Floor and Second Floor: Duplicate the walls from the Ground Floor
	first_floor_dividing_walls = duplicate_obj(wall_G_uuid, first_floor_uuid, 1) + \
								duplicate_obj(wall_H_uuid, first_floor_uuid, 1)

	second_floor_dividing_walls = duplicate_obj(wall_G_uuid, second_floor_uuid, 1) + \
								duplicate_obj(wall_H_uuid, second_floor_uuid, 1)

	# Step 5: Create Interior Walls for Apartments on Each Floor
	# Ground Floor
	# Apartment 1
	wall_I_uuid = create_wall((0, 5000), (15000, 5000), ground_floor_uuid)
	wall_L_uuid = create_wall((0, 10000), (15000, 10000), ground_floor_uuid)
	# Apartment 2
	wall_J_uuid = create_wall((15000, 5000), (30000, 5000), ground_floor_uuid)
	wall_M_uuid = create_wall((15000, 10000), (30000, 10000), ground_floor_uuid)
	# Apartment 3
	wall_K_uuid = create_wall((30000, 5000), (45000, 5000), ground_floor_uuid)
	wall_N_uuid = create_wall((30000, 10000), (45000, 10000), ground_floor_uuid)

	# First Floor and Second Floor: Duplicate the walls from the Ground Floor
	first_floor_interior_walls = duplicate_obj(wall_I_uuid, first_floor_uuid, 1) + \
								duplicate_obj(wall_L_uuid, first_floor_uuid, 1) + \
								duplicate_obj(wall_J_uuid, first_floor_uuid, 1) + \
								duplicate_obj(wall_M_uuid, first_floor_uuid, 1) + \
								duplicate_obj(wall_K_uuid, first_floor_uuid, 1) + \
								duplicate_obj(wall_N_uuid, first_floor_uuid, 1)

	second_floor_interior_walls = duplicate_obj(wall_I_uuid, second_floor_uuid, 1) + \
								duplicate_obj(wall_L_uuid, second_floor_uuid, 1) + \
								duplicate_obj(wall_J_uuid, second_floor_uuid, 1) + \
								duplicate_obj(wall_M_uuid, second_floor_uuid, 1) + \
								duplicate_obj(wall_K_uuid, second_floor_uuid, 1) + \
								duplicate_obj(wall_N_uuid, second_floor_uuid, 1)

	# Step 6: Add Main Entrance Doors for Each Apartment on Each Floor
	# Ground Floor
	door_apt1_uuid = add_door_to_wall(wall_A_uuid, 0, 20000, "Main Entrance Door")
	door_apt2_uuid = add_door_to_wall(wall_A_uuid, 0, 25000, "Main Entrance Door")
	door_apt3_uuid = add_door_to_wall(wall_C_uuid, 0, 20000, "Main Entrance Door")

	# First Floor and Second Floor: Duplicate the doors from the Ground Floor
	first_floor_doors = duplicate_obj(door_apt1_uuid, first_floor_uuid, 1) + \
					duplicate_obj(door_apt2_uuid, first_floor_uuid, 1) + \
					duplicate_obj(door_apt3_uuid, first_floor_uuid, 1)

	second_floor_doors = duplicate_obj(door_apt1_uuid, second_floor_uuid, 1) + \
						duplicate_obj(door_apt2_uuid, second_floor_uuid, 1) + \
						duplicate_obj(door_apt3_uuid, second_floor_uuid, 1)

	# Step 7: Add Interior Doors for Bedrooms, Bathrooms, and Kitchens
	# Ground Floor: Place doors appropriately within each apartment
	# (Assuming specific coordinates for interior doors)
	interior_door_1_uuid = add_door_to_wall(wall_I_uuid, 0, 7500, "Interior Door")
	interior_door_2_uuid = add_door_to_wall(wall_L_uuid, 0, 7500, "Interior Door")
	interior_door_3_uuid = add_door_to_wall(wall_J_uuid, 0, 22500, "Interior Door")
	interior_door_4_uuid = add_door_to_wall(wall_M_uuid, 0, 22500, "Interior Door")
	interior_door_5_uuid = add_door_to_wall(wall_K_uuid, 0, 37500, "Interior Door")
	interior_door_6_uuid = add_door_to_wall(wall_N_uuid, 0, 37500, "Interior Door")

	# First Floor and Second Floor: Duplicate the doors from the Ground Floor
	first_floor_interior_doors = duplicate_obj(interior_door_1_uuid, first_floor_uuid, 1) + \
								duplicate_obj(interior_door_2_uuid, first_floor_uuid, 1) + \
								duplicate_obj(interior_door_3_uuid, first_floor_uuid, 1) + \
								duplicate_obj(interior_door_4_uuid, first_floor_uuid, 1) + \
								duplicate_obj(interior_door_5_uuid, first_floor_uuid, 1) + \
								duplicate_obj(interior_door_6_uuid, first_floor_uuid, 1)

	second_floor_interior_doors = duplicate_obj(interior_door_1_uuid, second_floor_uuid, 1) + \
								duplicate_obj(interior_door_2_uuid, second_floor_uuid, 1) + \
								duplicate_obj(interior_door_3_uuid, second_floor_uuid, 1) + \
								duplicate_obj(interior_door_4_uuid, second_floor_uuid, 1) + \
								duplicate_obj(interior_door_5_uuid, second_floor_uuid, 1) + \
								duplicate_obj(interior_door_6_uuid, second_floor_uuid, 1)

	# Step 8: Add Windows for Natural Light
	# Ground Floor: Place windows on the exterior walls of each apartment
	# (Assuming specific coordinates for windows)
	window_1_uuid = add_window_to_wall(wall_A_uuid, 1500, 10000, "Window")
	window_2_uuid = add_window_to_wall(wall_B_uuid, 1500, 7500, "Window")
	window_3_uuid = add_window_to_wall(wall_C_uuid, 1500, 10000, "Window")
	window_4_uuid = add_window_to_wall(wall_D_uuid, 1500, 7500, "Window")

	# First Floor and Second Floor: Duplicate the windows from the Ground Floor
	first_floor_windows = duplicate_obj(window_1_uuid, first_floor_uuid, 1) + \
						duplicate_obj(window_2_uuid, first_floor_uuid, 1) + \
						duplicate_obj(window_3_uuid, first_floor_uuid, 1) + \
						duplicate_obj(window_4_uuid, first_floor_uuid, 1)

	second_floor_windows = duplicate_obj(window_1_uuid, second_floor_uuid, 1) + \
						duplicate_obj(window_2_uuid, second_floor_uuid, 1) + \
						duplicate_obj(window_3_uuid, second_floor_uuid, 1) + \
						duplicate_obj(window_4_uuid, second_floor_uuid, 1)

	# Step 9: Create Slabs for Each Floor
	# Ground Floor: Create slab covering entire floor area
	ground_floor_polygon_uuid = create_polygon([(0, 0), (45000, 0), (45000, 15000), (0, 15000)], ground_floor_uuid)
	ground_floor_slab_uuid = create_slab(ground_floor_polygon_uuid, ground_floor_uuid)

	# First Floor and Second Floor: Duplicate the slab from the Ground Floor
	first_floor_slab_uuid = duplicate_obj(ground_floor_slab_uuid, first_floor_uuid, 1)
	second_floor_slab_uuid = duplicate_obj(ground_floor_slab_uuid, second_floor_uuid, 1)

	# Step 10: Create Roof
	# Roof: Create a roof covering the entire building
	roof_polygon_uuid = create_polygon([(0, 0), (45000, 0), (45000, 15000), (0, 15000)], second_floor_uuid)
	roof_uuid = create_pitched_roof(roof_polygon_uuid, second_floor_uuid, slope=30, eave_overhang=500, eave_height=3000, roof_thickness=300)

def test_llm_code_custom_interpreter():
	code = """

# Phase 1: Setup & Ground Floor

# 1. Story Layers
ground_floor_layer_uuid = create_story_layer(layer_name="Ground Floor", elevation=0, floor_index=1)
first_floor_layer_uuid = create_story_layer(layer_name="First Floor", elevation=3000, floor_index=2)
roof_layer_uuid = create_story_layer(layer_name="Roof Layer", elevation=6000, floor_index=3)
set_active_story_layer(layer_name="Ground Floor")

# 2. Ground Floor Slab
gf_slab_poly_vertices = [(0,0), (15000,0), (15000,10000), (0,10000)]
gf_slab_poly_uuid = create_polygon(vertices=gf_slab_poly_vertices, layer_uuid=ground_floor_layer_uuid)
gf_slab_uuid = create_slab(profile_id=gf_slab_poly_uuid, layer_uuid=ground_floor_layer_uuid)
set_slab_height(slab_id=gf_slab_uuid, height=0)
set_slab_style(slab_id=gf_slab_uuid, style_name="Generic Concrete Slab") # Assuming "Generic Concrete Slab" is a valid style

# 3. Ground Floor Functional Areas
entrance_hall_vertices = [(0,0), (2500,0), (2500,4000), (0,4000)]
fa_entrance_hall_uuid = create_functional_area(vertices=entrance_hall_vertices, name="Entrance Hall", layer_uuid=ground_floor_layer_uuid)

wc_vertices = [(0,4000), (2500,4000), (2500,6000), (0,6000)]
fa_wc_uuid = create_functional_area(vertices=wc_vertices, name="WC", layer_uuid=ground_floor_layer_uuid)

living_room_vertices = [(2500,0), (9000,0), (9000,6000), (2500,6000)]
fa_living_room_uuid = create_functional_area(vertices=living_room_vertices, name="Living Room", layer_uuid=ground_floor_layer_uuid)

kitchen_dining_vertices = [(9000,0), (15000,0), (15000,6000), (9000,6000)]
fa_kitchen_dining_uuid = create_functional_area(vertices=kitchen_dining_vertices, name="Kitchen/Dining", layer_uuid=ground_floor_layer_uuid)

staircase_area_vertices = [(0,6000), (3000,6000), (3000,10000), (0,10000)]
fa_staircase_area_uuid = create_functional_area(vertices=staircase_area_vertices, name="Staircase Area", layer_uuid=ground_floor_layer_uuid)

utility_room_vertices = [(3000,6000), (7500,6000), (7500,10000), (3000,10000)]
fa_utility_room_uuid = create_functional_area(vertices=utility_room_vertices, name="Utility Room", layer_uuid=ground_floor_layer_uuid)

study_office_vertices = [(7500,6000), (15000,6000), (15000,10000), (7500,10000)]
fa_study_office_uuid = create_functional_area(vertices=study_office_vertices, name="Study/Office", layer_uuid=ground_floor_layer_uuid)

# 4. Ground Floor Walls
# Perimeter Walls
wall_A_gf_uuid = create_wall(st_pt=(0,0), ed_pt=(15000,0), layer_uuid=ground_floor_layer_uuid)
set_wall_thickness(uuid=wall_A_gf_uuid, thickness=200)
set_wall_style(uuid=wall_A_gf_uuid, style_name="Exterior Concrete Wall")

wall_B_gf_uuid = create_wall(st_pt=(15000,0), ed_pt=(15000,10000), layer_uuid=ground_floor_layer_uuid)
set_wall_thickness(uuid=wall_B_gf_uuid, thickness=200)
set_wall_style(uuid=wall_B_gf_uuid, style_name="Exterior Concrete Wall")

wall_C_gf_uuid = create_wall(st_pt=(15000,10000), ed_pt=(0,10000), layer_uuid=ground_floor_layer_uuid)
set_wall_thickness(uuid=wall_C_gf_uuid, thickness=200)
set_wall_style(uuid=wall_C_gf_uuid, style_name="Exterior Concrete Wall")

wall_D_gf_uuid = create_wall(st_pt=(0,10000), ed_pt=(0,0), layer_uuid=ground_floor_layer_uuid)
set_wall_thickness(uuid=wall_D_gf_uuid, thickness=200)
set_wall_style(uuid=wall_D_gf_uuid, style_name="Exterior Concrete Wall")

# Internal Walls
wall_G1_gf_uuid = create_wall(st_pt=(2500,0), ed_pt=(2500,4000), layer_uuid=ground_floor_layer_uuid)
set_wall_thickness(uuid=wall_G1_gf_uuid, thickness=100)
set_wall_style(uuid=wall_G1_gf_uuid, style_name="Interior Concrete Wall")

wall_G2_gf_uuid = create_wall(st_pt=(0,4000), ed_pt=(2500,4000), layer_uuid=ground_floor_layer_uuid)
set_wall_thickness(uuid=wall_G2_gf_uuid, thickness=100)
set_wall_style(uuid=wall_G2_gf_uuid, style_name="Interior Concrete Wall")

wall_G3_gf_uuid = create_wall(st_pt=(2500,4000), ed_pt=(2500,6000), layer_uuid=ground_floor_layer_uuid)
set_wall_thickness(uuid=wall_G3_gf_uuid, thickness=100)
set_wall_style(uuid=wall_G3_gf_uuid, style_name="Interior Concrete Wall")

wall_G4_gf_uuid = create_wall(st_pt=(9000,0), ed_pt=(9000,6000), layer_uuid=ground_floor_layer_uuid)
set_wall_thickness(uuid=wall_G4_gf_uuid, thickness=100)
set_wall_style(uuid=wall_G4_gf_uuid, style_name="Interior Concrete Wall")

wall_G5_gf_uuid = create_wall(st_pt=(0,6000), ed_pt=(15000,6000), layer_uuid=ground_floor_layer_uuid)
set_wall_thickness(uuid=wall_G5_gf_uuid, thickness=100)
set_wall_style(uuid=wall_G5_gf_uuid, style_name="Interior Concrete Wall")

wall_G6_gf_uuid = create_wall(st_pt=(3000,6000), ed_pt=(3000,10000), layer_uuid=ground_floor_layer_uuid)
set_wall_thickness(uuid=wall_G6_gf_uuid, thickness=100)
set_wall_style(uuid=wall_G6_gf_uuid, style_name="Interior Concrete Wall")

wall_G7_gf_uuid = create_wall(st_pt=(7500,6000), ed_pt=(7500,10000), layer_uuid=ground_floor_layer_uuid)
set_wall_thickness(uuid=wall_G7_gf_uuid, thickness=100)
set_wall_style(uuid=wall_G7_gf_uuid, style_name="Interior Concrete Wall")

# 5. Ground Floor Doors
door_gf1_uuid = add_door_to_wall(wall_uuid=wall_A_gf_uuid, door_elevation=0, door_offset=1000, door_name="Standard Door")
door_gf2_uuid = add_door_to_wall(wall_uuid=wall_G2_gf_uuid, door_elevation=0, door_offset=1250, door_name="Standard Door")
door_gf3_uuid = add_door_to_wall(wall_uuid=wall_G1_gf_uuid, door_elevation=0, door_offset=1000, door_name="Standard Door") # Offset from (2500,0)
door_gf4_uuid = add_door_to_wall(wall_uuid=wall_G4_gf_uuid, door_elevation=0, door_offset=2500, door_name="Standard Door") # Offset from (9000,0)
door_gf5_uuid = add_door_to_wall(wall_uuid=wall_G5_gf_uuid, door_elevation=0, door_offset=1000, door_name="Standard Door") # Offset from (0,6000)
door_gf6_uuid = add_door_to_wall(wall_uuid=wall_G6_gf_uuid, door_elevation=0, door_offset=1000, door_name="Standard Door") # Offset from (3000,6000)
door_gf7_uuid = add_door_to_wall(wall_uuid=wall_G5_gf_uuid, door_elevation=0, door_offset=8500, door_name="Standard Door") # Offset from (0,6000)

# 6. Ground Floor Windows
window_gf1_uuid = add_window_to_wall(wall_uuid=wall_A_gf_uuid, window_elevation=900, window_offset=4500, window_name="Standard Window")
window_gf2_uuid = add_window_to_wall(wall_uuid=wall_A_gf_uuid, window_elevation=900, window_offset=10500, window_name="Standard Window")
window_gf3_uuid = add_window_to_wall(wall_uuid=wall_B_gf_uuid, window_elevation=900, window_offset=2000, window_name="Standard Window")
window_gf4_uuid = add_window_to_wall(wall_uuid=wall_D_gf_uuid, window_elevation=900, window_offset=4700, window_name="Standard Window") # Wall D (0,10000) to (0,0)
window_gf5_uuid = add_window_to_wall(wall_uuid=wall_C_gf_uuid, window_elevation=900, window_offset=9000, window_name="Standard Window") # Wall C (15000,10000) to (0,10000)
window_gf6_uuid = add_window_to_wall(wall_uuid=wall_C_gf_uuid, window_elevation=900, window_offset=2500, window_name="Standard Window") # Wall C (15000,10000) to (0,10000)
window_gf7_uuid = add_window_to_wall(wall_uuid=wall_B_gf_uuid, window_elevation=900, window_offset=7000, window_name="Standard Window")

# Phase 2: First Floor

# 1. Set "First Floor" as active layer.
set_active_story_layer(layer_name="First Floor")

# 2. First Floor Slab (with Balconies)
ff_main_slab_poly_vertices = [(0,0), (15000,0), (15000,10000), (0,10000)]
ff_main_slab_poly_uuid = create_polygon(vertices=ff_main_slab_poly_vertices, layer_uuid=first_floor_layer_uuid)
ff_main_slab_uuid = create_slab(profile_id=ff_main_slab_poly_uuid, layer_uuid=first_floor_layer_uuid)
set_slab_height(slab_id=ff_main_slab_uuid, height=0)
set_slab_style(slab_id=ff_main_slab_uuid, style_name="Generic Concrete Slab")

ff_front_balcony_poly_vertices = [(0,-1500), (15000,-1500), (15000,0), (0,0)]
ff_front_balcony_poly_uuid = create_polygon(vertices=ff_front_balcony_poly_vertices, layer_uuid=first_floor_layer_uuid)
ff_front_balcony_slab_uuid = create_slab(profile_id=ff_front_balcony_poly_uuid, layer_uuid=first_floor_layer_uuid)
set_slab_height(slab_id=ff_front_balcony_slab_uuid, height=0)
set_slab_style(slab_id=ff_front_balcony_slab_uuid, style_name="Generic Concrete Slab")

ff_rear_balcony_poly_vertices = [(0,10000), (15000,10000), (15000,11500), (0,11500)]
ff_rear_balcony_poly_uuid = create_polygon(vertices=ff_rear_balcony_poly_vertices, layer_uuid=first_floor_layer_uuid)
ff_rear_balcony_slab_uuid = create_slab(profile_id=ff_rear_balcony_poly_uuid, layer_uuid=first_floor_layer_uuid)
set_slab_height(slab_id=ff_rear_balcony_slab_uuid, height=0)
set_slab_style(slab_id=ff_rear_balcony_slab_uuid, style_name="Generic Concrete Slab")

# 3. First Floor Functional Areas
landing_corridor_vertices = [(0,6000), (15000,6000), (15000,7000), (0,7000)]
fa_landing_corridor_uuid = create_functional_area(vertices=landing_corridor_vertices, name="Landing/Corridor", layer_uuid=first_floor_layer_uuid)

master_bedroom_vertices = [(0,0), (7500,0), (7500,6000), (0,6000)]
fa_master_bedroom_uuid = create_functional_area(vertices=master_bedroom_vertices, name="Master Bedroom", layer_uuid=first_floor_layer_uuid)

ensuite_vertices = [(0,4500), (2500,4500), (2500,6000), (0,6000)] # Corrected based on wall F3a, F3b
fa_ensuite_uuid = create_functional_area(vertices=ensuite_vertices, name="En-suite", layer_uuid=first_floor_layer_uuid)

bedroom2_vertices = [(7500,0), (15000,0), (15000,6000), (7500,6000)]
fa_bedroom2_uuid = create_functional_area(vertices=bedroom2_vertices, name="Bedroom 2", layer_uuid=first_floor_layer_uuid)

family_bathroom_vertices = [(0,7000), (4000,7000), (4000,10000), (0,10000)]
fa_family_bathroom_uuid = create_functional_area(vertices=family_bathroom_vertices, name="Family Bathroom", layer_uuid=first_floor_layer_uuid)

bedroom3_vertices = [(4000,7000), (9500,7000), (9500,10000), (4000,10000)]
fa_bedroom3_uuid = create_functional_area(vertices=bedroom3_vertices, name="Bedroom 3", layer_uuid=first_floor_layer_uuid)

bedroom4_vertices = [(9500,7000), (15000,7000), (15000,10000), (9500,10000)]
fa_bedroom4_uuid = create_functional_area(vertices=bedroom4_vertices, name="Bedroom 4", layer_uuid=first_floor_layer_uuid)

# 4. First Floor Walls
# Perimeter Walls
wall_FA_ff_uuid = create_wall(st_pt=(0,0), ed_pt=(15000,0), layer_uuid=first_floor_layer_uuid)
set_wall_thickness(uuid=wall_FA_ff_uuid, thickness=200)
set_wall_style(uuid=wall_FA_ff_uuid, style_name="Exterior Concrete Wall")

wall_FB_ff_uuid = create_wall(st_pt=(15000,0), ed_pt=(15000,10000), layer_uuid=first_floor_layer_uuid)
set_wall_thickness(uuid=wall_FB_ff_uuid, thickness=200)
set_wall_style(uuid=wall_FB_ff_uuid, style_name="Exterior Concrete Wall")

wall_FC_ff_uuid = create_wall(st_pt=(15000,10000), ed_pt=(0,10000), layer_uuid=first_floor_layer_uuid)
set_wall_thickness(uuid=wall_FC_ff_uuid, thickness=200)
set_wall_style(uuid=wall_FC_ff_uuid, style_name="Exterior Concrete Wall")

wall_FD_ff_uuid = create_wall(st_pt=(0,10000), ed_pt=(0,0), layer_uuid=first_floor_layer_uuid)
set_wall_thickness(uuid=wall_FD_ff_uuid, thickness=200)
set_wall_style(uuid=wall_FD_ff_uuid, style_name="Exterior Concrete Wall")

# Internal Walls
wall_F1_ff_uuid = create_wall(st_pt=(0,6000), ed_pt=(15000,6000), layer_uuid=first_floor_layer_uuid)
set_wall_thickness(uuid=wall_F1_ff_uuid, thickness=100)
set_wall_style(uuid=wall_F1_ff_uuid, style_name="Interior Concrete Wall")

wall_F2_ff_uuid = create_wall(st_pt=(0,7000), ed_pt=(15000,7000), layer_uuid=first_floor_layer_uuid)
set_wall_thickness(uuid=wall_F2_ff_uuid, thickness=100)
set_wall_style(uuid=wall_F2_ff_uuid, style_name="Interior Concrete Wall")

wall_F3a_ff_uuid = create_wall(st_pt=(0,4500), ed_pt=(2500,4500), layer_uuid=first_floor_layer_uuid)
set_wall_thickness(uuid=wall_F3a_ff_uuid, thickness=100)
set_wall_style(uuid=wall_F3a_ff_uuid, style_name="Interior Concrete Wall")

wall_F3b_ff_uuid = create_wall(st_pt=(2500,4500), ed_pt=(2500,6000), layer_uuid=first_floor_layer_uuid)
set_wall_thickness(uuid=wall_F3b_ff_uuid, thickness=100)
set_wall_style(uuid=wall_F3b_ff_uuid, style_name="Interior Concrete Wall")

wall_F4_ff_uuid = create_wall(st_pt=(7500,0), ed_pt=(7500,6000), layer_uuid=first_floor_layer_uuid)
set_wall_thickness(uuid=wall_F4_ff_uuid, thickness=100)
set_wall_style(uuid=wall_F4_ff_uuid, style_name="Interior Concrete Wall")

wall_F5_ff_uuid = create_wall(st_pt=(4000,7000), ed_pt=(4000,10000), layer_uuid=first_floor_layer_uuid)
set_wall_thickness(uuid=wall_F5_ff_uuid, thickness=100)
set_wall_style(uuid=wall_F5_ff_uuid, style_name="Interior Concrete Wall")

wall_F6_ff_uuid = create_wall(st_pt=(9500,7000), ed_pt=(9500,10000), layer_uuid=first_floor_layer_uuid)
set_wall_thickness(uuid=wall_F6_ff_uuid, thickness=100)
set_wall_style(uuid=wall_F6_ff_uuid, style_name="Interior Concrete Wall")

wall_F7_ff_uuid = create_wall(st_pt=(3000,6000), ed_pt=(3000,7000), layer_uuid=first_floor_layer_uuid)
set_wall_thickness(uuid=wall_F7_ff_uuid, thickness=100)
set_wall_style(uuid=wall_F7_ff_uuid, style_name="Interior Concrete Wall")

# 5. First Floor Doors
door_ff1_uuid = add_door_to_wall(wall_uuid=wall_F1_ff_uuid, door_elevation=0, door_offset=3750, door_name="Standard Door") # Offset from (0,6000)
door_ff2_uuid = add_door_to_wall(wall_uuid=wall_F3a_ff_uuid, door_elevation=0, door_offset=1250, door_name="Standard Door") # Offset from (0,4500)
door_ff3_uuid = add_door_to_wall(wall_uuid=wall_F1_ff_uuid, door_elevation=0, door_offset=8500, door_name="Standard Door") # Offset from (0,6000)
door_ff4_uuid = add_door_to_wall(wall_uuid=wall_F2_ff_uuid, door_elevation=0, door_offset=1000, door_name="Standard Door") # Offset from (0,7000)
# door_ff5_uuid = add_door_to_wall(wall_uuid=wall_F2_ff_uuid, door_elevation=0, door_offset=5000, do

"""
	tools = {}  
	for name, attr in vars(builtins).items():
		if callable(attr):
			tools[name] = attr

	# init tools
	create_wall_tool = CreateWallTool()
	set_wall_thickness_tool = SetWallThickness()
	set_wall_height_tool = SetWallHeight()
	set_wall_style_tool = SetWallStyle()
	get_wall_elevation_tool = GetWallElevation()
	get_wall_thickness_tool = GetWallThickness()
	add_window_tool = AddWindowToWall()
	add_door_tool = AddDoorToWall()
	move_tool = Move()
	delete_tool = DeleteTool()
	get_selected_tool = FindSelect()
	create_polygon_tool = CreatePolygon()
	get_polygon_vertex_tool = GetPolygonVertex()
	get_polygon_vertex_count_tool = GetVertNum()
	create_slab_tool = CreateSlab()
	set_slab_height_tool = SetSlabHeight()
	get_slab_height_tool = GetSlabHeight()
	set_slab_style_tool = SetSlabStyle()
	duplicate_object_tool = DuplicateObj()
	rotate_object_tool = RotateObj()
	create_roof_tool = CreateRoof()
	set_roof_attributes_tool = SetRoofAttributes()
	set_roof_style_tool = SetRoofStyle()
	create_story_layer = CreateStoryLayer()
	set_active_story_layer = SetStoryLayerActive()
	create_space = CreateSpace()

	additional_tools = [
		create_wall_tool, 
		set_wall_height_tool, 
		set_wall_thickness_tool, 
		set_wall_style_tool, 
		get_wall_elevation_tool, 
		get_wall_thickness_tool, 
		add_window_tool, 
		add_door_tool, 
		move_tool, 
		delete_tool, 
		get_selected_tool, 
		create_polygon_tool, 
		get_polygon_vertex_tool, 
		get_polygon_vertex_count_tool, 
		create_slab_tool, 
		set_slab_height_tool, 
		get_slab_height_tool,
		set_slab_style_tool, 
		duplicate_object_tool, 
		rotate_object_tool, 
		create_roof_tool, 
		set_roof_attributes_tool,
		set_roof_style_tool,
		create_story_layer,
		set_active_story_layer,
		create_space
	]
	if isinstance(additional_tools, (list, tuple)):
		additional_tools = {t.name: t for t in additional_tools}
	elif not isinstance(additional_tools, dict):
		additional_tools = {additional_tools.name: additional_tools}

	tools.update(additional_tools)

	result, state = evaluate(code, tools)

# create a command plugin in VW and run this function to debug the LLM code
def test_interface():
	# debug attach
	DEBUG = True
	if DEBUG:
		server = DebugServer.getInstance()

	import atexit

	atexit.register(server.stop)

	try:
		test_llm_code_custom_interpreter()
		# test_llm_code()
	except KeyboardInterrupt:
		server.stop()
		print("Server stopped.")
