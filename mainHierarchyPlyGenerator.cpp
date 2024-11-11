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
		throw std::runtime_error("Failed to pass args <plyfile> <outputpath>");

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

	std::cout << "Generating" << std::endl;

	PointbasedKdTreeGenerator generator;
	auto root = generator.generate(gaussians);

	std::cout << "Merging" << std::endl;

	ClusterMerger merger;
	merger.merge(root, gaussians);

	std::cout << "Fixing rotations" << std::endl;
	RotationAligner::align(root, gaussians);

	std::string outputpath(argv[2]);
	std::vector<Eigen::Vector3f> positions;
	std::vector<Eigen::Vector4f> rotations;
	std::vector<Eigen::Vector3f> log_scales;
	std::vector<float> opacities;
	std::vector<SHs> shs;
	std::vector<Node> basenodes;
	std::vector<Box> boxes;
	Writer::makeHierarchy(gaussians, root, positions, rotations, log_scales, opacities, shs, basenodes, boxes);
	gaussians.clear();

	// parent_id, count_leafs, node_size, parent_node_size
	std::vector<Eigen::Vector4i> hiers;
	hiers.resize(positions.size(), Eigen::Vector4i(0, 0, 0, 0));
	for (size_t i = 0; i < basenodes.size(); i++) {
		const Node& node = basenodes[i];
		const Box bbox = boxes[i];
		Eigen::Vector4i& hier_out = hiers[node.start];
		hier_out[1] = node.count_leafs;
		hier_out[2] = int(bbox.maxx[3] * 1000); // scale 1000
		if (node.parent >= 0) {
			hier_out[0] = basenodes[node.parent].start; // parent_index
			const Box& parent_bbox = boxes[node.parent];
			hier_out[3] = int(parent_bbox.maxx[3] * 1000); // scale 1000
		}
	}
	Writer::writePlyHierarchy(outputpath.c_str(), positions, rotations, log_scales, opacities, shs, hiers);
}