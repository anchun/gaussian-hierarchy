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
#include <vector>
#include <iostream>
#include "FlatGenerator.h"
#include "PointbasedKdTreeGenerator.h"
#include "AvgMerger.h"
#include "ClusterMerger.h"
#include "common.h"
#include <fstream>
#include <filesystem>
#include "appearance_filter.h"
#include "rotation_aligner.h"

void recTraverse(ExplicitTreeNode* node, int& zerocount)
{
	if (node->depth == 0)
		zerocount++;
	if (node->children.size() > 0 && node->depth == 0)
		throw std::runtime_error("Leaf nodes should never have children!");

	for (auto c : node->children)
	{
		recTraverse(c, zerocount);
	}
}

int main(int argc, char* argv[])
{
	if (argc < 3)
		throw std::runtime_error("Failed to pass args <plyfile> <source dir> [scaffold dir]");

	int skyboxpoints = 0;
	if (argc > 4)
	{
		const char* scaffoldpath = nullptr;
		scaffoldpath = argv[4];
		std::ifstream scaffoldfile(std::string(scaffoldpath) + "/pc_info.txt");

		std::string line;
		std::getline(scaffoldfile, line);
		skyboxpoints = std::atoi(line.c_str());
	}

	std::vector<Gaussian> gaussians_unfiltered;
	try
	{
		Loader::loadPly(argv[1], gaussians_unfiltered, skyboxpoints);
	}
	catch(const std::runtime_error& e)
	{
		std::cout << "Could not load .ply. Attempt loading .bin\n";
		std::string filename(argv[1]);
		filename.pop_back();  
		filename.pop_back();
		filename.pop_back();
		filename = filename + "bin";
		std::cout << filename << std::endl;
		Loader::loadBin(filename.c_str(), gaussians_unfiltered, skyboxpoints);
	}

	int valid = 0;
	bool not_warned[8] = { true };
	std::vector<Gaussian> gaussians(gaussians_unfiltered.size());
	for (int i = 0; i < gaussians_unfiltered.size(); i++)
	{
		Gaussian& g = gaussians_unfiltered[i];
		if (std::isinf(g.opacity))
		{
			if (not_warned[0])
				std::cout << "Found Inf opacity";
			not_warned[0] = false;
			continue;
		}
		if (std::isnan(g.opacity))
		{
			if (not_warned[1])
				std::cout << "Found NaN opacity";
			not_warned[1] = false;
			continue;
		}
		if (g.scale.hasNaN())
		{
			if (not_warned[2])
				std::cout << "Found NaN scale";
			not_warned[2] = false;
			continue;
		}
		if ((g.scale.x() < 1e-7f && g.scale.y() < 1e-7f) ||
			(g.scale.y() < 1e-7f && g.scale.z() < 1e-7f) ||
			(g.scale.x() < 1e-7f && g.scale.z() < 1e-7f))
		{
			if (not_warned[3])
				std::cout << "Found invalid scale";
			not_warned[3] = false;
			continue;
		}
		if (g.rotation.hasNaN())
		{
			if (not_warned[4])
				std::cout << "Found NaN rot";
			not_warned[4] = false;
			continue;
		}
		if (g.position.hasNaN())
		{
			if (not_warned[5])
				std::cout << "Found NaN pos";
			not_warned[5] = false;
			continue;
		}
		if (g.shs.hasNaN())
		{
			if(not_warned[6])
				std::cout << "Found NaN sh";
			not_warned[6] = false;
			continue;
		}
		if (g.opacity < 1e-7f)
		{
			if (not_warned[7])
				std::cout << "Found 0 opacity";
			not_warned[7] = false;
			continue;
		}

		gaussians[valid] = g;
		valid++;
	}

	gaussians.resize(valid);

	std::cout << "Generating" << std::endl;

	PointbasedKdTreeGenerator generator;
	auto root = generator.generate(gaussians);

	std::cout << "Merging" << std::endl;

	ClusterMerger merger;
	merger.merge(root, gaussians);

	std::cout << "Fixing rotations" << std::endl;
	RotationAligner::align(root, gaussians);

	std::cout << "Filtering" << std::endl;
	float limit = 0.0005f;

	char* filterpath = "";
	filterpath = argv[2];

	AppearanceFilter filter;
	filter.init(filterpath);
	filter.filter(root, gaussians, limit, 2.0f);

	filter.writeAnchors((std::string(argv[3]) + "/anchors.bin").c_str(), root, gaussians, limit);
	
	std::cout << "Writing" << std::endl;

	Writer::writeHierarchy((std::string(argv[3]) + "/hierarchy.hier").c_str(), gaussians, root, true);
}