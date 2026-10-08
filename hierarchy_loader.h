/*
 * Copyright (C) 2024, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include "types.h"

#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <string>

class HierarchyLoader
{
public:
	static void load(const char* filename,
		std::vector<Eigen::Vector3f>& pos,
		std::vector<SHs>& shs,
		std::vector<float>& alphas,
		std::vector<Eigen::Vector3f>& scales,
		std::vector<Eigen::Vector4f>& rot,
		std::vector<Node>& nodes,
		std::vector<Box>& boxes);

	static bool loadAdaptiveUnits(const char* filename,
		float target_extent,
		std::vector<int>& node_ids,
		std::vector<Box>& unit_boxes,
		int& selected_depth,
		std::string& error,
		int requested_depth = -1);
	
};