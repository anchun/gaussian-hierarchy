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
	if (argc < 5)
		throw std::runtime_error("Failed to pass filename");

	int chunk_count(argc - 5);
	std::string rootpath(argv[1]);
	std::string outputpath(argv[4]);
	int with_hierarchy = std::stoi(argv[2]);
	//bool writeSky = false;
	{
		// Read chunk centers
		std::string inpath(argv[3]);
		std::vector<Eigen::Vector3f> chunk_centers(chunk_count);
		for (int chunk_id(0); chunk_id < chunk_count; chunk_id++)
		{
			int argidx(chunk_id + 5);
			std::ifstream f(inpath + "/" + argv[argidx] + "/center.txt");
			Eigen::Vector3f chunk_center(0.f, 0.f, 0.f);
			f >> chunk_center[0]; f >> chunk_center[1]; f >> chunk_center[2];
			chunk_centers[chunk_id] = chunk_center;
		}

		// Read per chunk hierarchies and discard unwanted primitives 
		// based on the distance to the chunk's center
		std::vector<Gaussian> gaussians; 
		ExplicitTreeNode* root = new ExplicitTreeNode;

		for (int chunk_id(0); chunk_id < chunk_count; chunk_id++)
		{
			int argidx(chunk_id + 5);
			std::cout << "Adding hierarchy for chunk " << argv[argidx] << std::endl;
			
			ExplicitTreeNode* chunkRoot = new ExplicitTreeNode;
			HierarchyExplicitLoader::loadExplicit(
				(rootpath + "/" + argv[argidx] + "/hierarchy.hier_opt").c_str(),
				gaussians, chunkRoot, chunk_id, chunk_centers);

			if (chunk_id == 0)
			{
				root->bounds = chunkRoot->bounds;
			}
			else
			{	
				for (int idx(0); idx < 3; idx++)
				{
					root->bounds.minn[idx] = std::min(root->bounds.minn[idx], chunkRoot->bounds.minn[idx]);
					root->bounds.maxx[idx] = std::max(root->bounds.maxx[idx], chunkRoot->bounds.maxx[idx]);
				}
			}
			root->depth = std::max(root->depth, chunkRoot->depth + 1);
			root->children.push_back(chunkRoot);
			root->merged.push_back(chunkRoot->merged[0]);
			root->bounds.maxx[3] = 1e9f;
			root->bounds.minn[3] = 1e9f;
		}
		if (chunk_count > 1) {
			Gaussian gaussian = AvgMerger::mergeGaussians(root->merged);
			root->merged.clear();
			root->merged.emplace_back(gaussian);
		}
		// 
		std::vector<Eigen::Vector3f> positions;
		std::vector<Eigen::Vector4f> rotations;
		std::vector<Eigen::Vector3f> log_scales;
		std::vector<float> opacities;
		std::vector<SHs> shs;
		std::vector<Node> basenodes;
		std::vector<Box> boxes;
		Writer::makeHierarchy(gaussians, root, positions, rotations, log_scales, opacities, shs, basenodes, boxes);

		std::string ext = outputpath.substr(outputpath.size() - 4);
		if (ext != ".ply") {
			HierarchyWriter writer;
			writer.write(
				outputpath.c_str(),
				positions.size(),
				basenodes.size(),
				positions.data(),
				shs.data(),
				opacities.data(),
				log_scales.data(),
				rotations.data(),
				basenodes.data(),
				boxes.data(),
				true
			);
		}
		else {
			if (with_hierarchy) {
				std::vector<Eigen::Vector4i> hiers;
				std::vector<Eigen::Vector4f> bboxes;
				hiers.resize(positions.size(), Eigen::Vector4i(0, -1, 0, 0));
				bboxes.resize(positions.size(), Eigen::Vector4f(0, 0, 0, 0));
				for (size_t i = 0; i < basenodes.size(); i++) {
					const Node& node = basenodes[i];
					const Box bbox = boxes[i];
					Eigen::Vector4i& hier_out = hiers[node.start];
					Eigen::Vector4f& bbox_out = bboxes[node.start];
					hier_out[0] = node.depth;
					hier_out[2] = node.count_leafs;
					hier_out[3] = node.count_children;
					bbox_out[0] = bbox.minn[3];
					bbox_out[1] = bbox.maxx[3];
					if (node.parent >= 0) {
						hier_out[1] = basenodes[node.parent].start; // parent_index
						const Box& parent_bbox = boxes[node.parent];
						bbox_out[2] = parent_bbox.minn[3];
						bbox_out[3] = parent_bbox.maxx[3];
					}
				}
				Writer::writePlyHierarchy(outputpath.c_str(), positions, rotations, log_scales, opacities, shs, hiers, bboxes);
			}
			else {
				Writer::writePly(outputpath.c_str(), gaussians, 0);
			}
		}
	}
}