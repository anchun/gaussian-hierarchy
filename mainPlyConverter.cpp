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


#include "loader.h"
#include "writer.h"
#include "FlatGenerator.h"
#include "PointbasedKdTreeGenerator.h"
#include "AvgMerger.h"
#include "ClusterMerger.h"
#include "common.h"
#include "dependencies/json.hpp"
#include "hierarchy_explicit_loader.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "appearance_filter.h"
#include "rotation_aligner.h"
#include "hierarchy_writer.h"

using json = nlohmann::json;

int main(int argc, char* argv[])
{
	if (argc < 3)
		throw std::runtime_error("Failed to pass args <plyfile> <outputpath> <degree>");

	std::vector<Gaussian> gaussians;
	try
	{
		Loader::loadPly(argv[1], gaussians, 0);
	}
	catch (const std::runtime_error& e)
	{
		std::cout << "Could not load .ply. Attempt loading .bin\n";
		std::string filename(argv[1]);
		filename.pop_back();
		filename.pop_back();
		filename.pop_back();
		filename = filename + "bin";
		std::cout << filename << std::endl;
		Loader::loadBin(filename.c_str(), gaussians, 0);
	}

	std::string outputpath(argv[2]);
	int sh_degree = 0;
	if(argc > 3)
		sh_degree = std::stoi(argv[3]);
	Writer::writePly(outputpath.c_str(), gaussians, sh_degree);
}