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
#include "hierarchy_explicit_loader.h"
#include "hierarchy_loader.h"
#include "merger_policy.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "appearance_filter.h"
#include "rotation_aligner.h"
#include "hierarchy_writer.h"
#include <algorithm>

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
	if (argc < 6)
		throw std::runtime_error("Expected trained root, SH degree, source root, output and at least one chunk");

	int chunk_count(argc - 5);
	std::string rootpath(argv[1]);
	std::string outputpath(argv[4]);
	int sh_degree = std::stoi(argv[2]);
	const bool write_ply = outputpath.size() >= 4
		&& outputpath.substr(outputpath.size() - 4) == ".ply";
	const std::string inpath(argv[3]);
	std::vector<std::string> source_directories, trained_directories;
	for (int chunk = 0; chunk < chunk_count; ++chunk)
	{
		source_directories.push_back(inpath + "/" + argv[chunk + 5]);
		trained_directories.push_back(rootpath + "/" + argv[chunk + 5]);
	}
	std::vector<SpatialChunkInfo> spatial_infos;
	bool spatial_ownership = false;
	std::string spatial_error;
	if (!validateSpatialChunkSet(source_directories, trained_directories,
		spatial_infos, spatial_ownership, spatial_error))
		throw std::runtime_error(spatial_error);
	if (!spatial_ownership)
		std::cout << "No spatial metadata: preserving legacy/corridor merge policy" << std::endl;
	{
		std::vector<Gaussian> scaffold_gaussians;
		if (write_ply)
		{
			const std::string scaffold_path = rootpath + "/../scaffold/point_cloud/iteration_30000";
			std::ifstream scaffoldfile(scaffold_path + "/pc_info.txt");
			if (!scaffoldfile.good())
				throw std::runtime_error("Global scaffold pc_info.txt not found");
			std::string line;
			std::getline(scaffoldfile, line);
			int scaffold_points = std::atoi(line.c_str());
			Loader::loadPly((scaffold_path + "/point_cloud.ply").c_str(), scaffold_gaussians);
			if(scaffold_gaussians.size() > scaffold_points)
				scaffold_gaussians.resize(scaffold_points);
			std::cout << "Global scaffold count: " << scaffold_gaussians.size() << std::endl;

			PriorInfo prior_info;
			std::string prior_error;
			const std::string prior_file = scaffold_path + "/prior_info.json";
			if (loadPriorInfo(prior_file, scaffold_gaussians.size(), prior_info, prior_error))
			{
				const SkyRepairStats stats = repairAnomalousSky(scaffold_gaussians, prior_info);
				std::cout << "Automatic sky repair: " << stats.repaired_points << "/"
					<< stats.sky_points << " points, minimum luminance threshold "
					<< stats.min_luminance_threshold << ", directional amplitude threshold "
					<< stats.directional_amplitude_threshold << std::endl;
			}
			else
			{
				std::cerr << "Warning: automatic sky repair disabled: " << prior_error << std::endl;
			}
		}

		// Read chunk centers
		std::vector<Eigen::Vector3f> chunk_centers(chunk_count);
		for (int chunk_id(0); chunk_id < chunk_count; chunk_id++)
		{
			int argidx(chunk_id + 5);
			std::ifstream f(inpath + "/" + argv[argidx] + "/center.txt");
			Eigen::Vector3f chunk_center(0.f, 0.f, 0.f);
			f >> chunk_center[0]; f >> chunk_center[1]; f >> chunk_center[2];
			chunk_centers[chunk_id] = chunk_center;
		}
		std::vector<std::string> hierarchy_paths(chunk_count);
		for (int chunk_id(0); chunk_id < chunk_count; chunk_id++)
		{
			const int argidx = chunk_id + 5;
			hierarchy_paths[chunk_id] = rootpath + "/" + argv[argidx] + "/hierarchy.hier_opt";
			std::ifstream hierarchy_file(hierarchy_paths[chunk_id], std::ios_base::binary);
			if (!hierarchy_file.good() || hierarchy_file.peek() == std::ifstream::traits_type::eof())
				hierarchy_paths[chunk_id] = rootpath + "/" + argv[argidx] + "/hierarchy.hier";
		}

		bool route_ownership = false;
		RouteOwnershipPlan ownership_plan;
		SpatialOwnershipPlan spatial_plan;
		if (spatial_ownership)
		{
			std::vector<std::vector<OwnershipUnit>> units(chunk_count);
			std::vector<std::vector<CameraSupport>> cameras(chunk_count);
			const float extent = spatialOwnershipExtent(spatial_infos.front());
			for (int chunk = 0; chunk < chunk_count; ++chunk)
			{
				std::vector<int> ids;
				std::vector<Box> boxes;
				if (!HierarchyLoader::loadSpatialFrontier(hierarchy_paths[chunk].c_str(),
					extent, ids, boxes, spatial_error)
					|| !loadSpatialCameraSupport(trained_directories[chunk] + "/cameras.json",
						spatial_infos[chunk], cameras[chunk], spatial_error))
					throw std::runtime_error(spatial_error);
				for (std::size_t index = 0; index < ids.size(); ++index)
					units[chunk].push_back({ids[index], boxes[index]});
			}
			spatial_plan = buildSpatialOwnership(units, spatial_infos, cameras);
			std::cout << "Spatial ownership: " << chunk_count << " chunks, "
				<< spatial_plan.stats.units << " complete-frontier units, "
				<< spatial_plan.stats.matched_pairs << " reciprocal pairs, "
				<< spatial_plan.stats.excluded_units << " directly matched copies excluded, "
				<< spatial_plan.stats.unmatched_units << " unmatched units retained (100%), "
				<< spatial_plan.stats.uncertain_units << " uncertain-size/invalid-box units retained"
				<< "; no center-distance opacity falloff" << std::endl;
		}
		else if (chunk_count == 2)
		{
			ChunkRouteInfo routes[2];
			std::vector<CameraSupport> cameras[2];
			RouteTransition transition;
			std::string route_error;
			bool route_valid = true;
			for (int chunk_id = 0; chunk_id < 2 && route_valid; ++chunk_id)
			{
				const int argidx = chunk_id + 5;
				route_valid = loadChunkRouteInfo(
					inpath + "/" + argv[argidx] + "/chunk_info.json", routes[chunk_id], route_error);
				if (route_valid && routes[chunk_id].ambiguous)
				{
					route_error = "route metadata is ambiguous";
					route_valid = false;
				}
				if (route_valid && !routes[chunk_id].explicitly_requested)
				{
					route_error = "route ownership requires explicitly requested corridor mode";
					route_valid = false;
				}
				if (route_valid)
					route_valid = loadCameraSupport(
						rootpath + "/" + argv[argidx] + "/cameras.json", routes[chunk_id], cameras[chunk_id], route_error);
			}
			if (route_valid)
				route_valid = deriveRouteTransition(routes[0], routes[1], transition, route_error);
			std::vector<OwnershipUnit> units[2];
			int selected_depth[2] = {-1, -1};
			if (route_valid)
			{
				const float target_extent = std::max(1.f, (transition.end - transition.start) / 10.f);
				for (int chunk_id = 0; chunk_id < 2 && route_valid; ++chunk_id)
				{
					std::vector<int> node_ids;
					std::vector<Box> boxes;
					route_valid = HierarchyLoader::loadAdaptiveUnits(
						hierarchy_paths[chunk_id].c_str(), target_extent,
						node_ids, boxes, selected_depth[chunk_id], route_error);
					if (route_valid)
					{
						units[chunk_id].reserve(node_ids.size());
						for (std::size_t index = 0; index < node_ids.size(); ++index)
							units[chunk_id].push_back({node_ids[index], boxes[index]});
					}
				}
				if (route_valid && selected_depth[0] != selected_depth[1])
				{
					const int common_depth = std::max(selected_depth[0], selected_depth[1]);
					for (int chunk_id = 0; chunk_id < 2 && route_valid; ++chunk_id)
					{
						if (selected_depth[chunk_id] == common_depth)
							continue;
						std::vector<int> node_ids;
						std::vector<Box> boxes;
						units[chunk_id].clear();
						route_valid = HierarchyLoader::loadAdaptiveUnits(
							hierarchy_paths[chunk_id].c_str(), target_extent,
							node_ids, boxes, selected_depth[chunk_id], route_error, common_depth);
						for (std::size_t index = 0; route_valid && index < node_ids.size(); ++index)
							units[chunk_id].push_back({node_ids[index], boxes[index]});
					}
				}
			}
			if (route_valid)
			{
				ownership_plan = buildRouteOwnership(
					units[0], units[1], routes[0], routes[1], transition, cameras[0], cameras[1]);
				route_ownership = ownership_plan.stats.matched_pairs > 0;
				std::cout << "Automatic route ownership: seam " << transition.seam
					<< ", band [" << transition.start << ", " << transition.end << "]"
					<< ", hierarchy depths " << selected_depth[0] << "/" << selected_depth[1]
					<< ", sampled cameras " << cameras[0].size() << "/" << cameras[1].size()
					<< ", matched pairs " << ownership_plan.stats.matched_pairs
					<< " (before/in/after " << ownership_plan.stats.matched_before_transition
					<< "/" << ownership_plan.stats.matched_in_transition << "/"
					<< ownership_plan.stats.matched_after_transition << ")"
					<< ", camera-arbitrated " << ownership_plan.stats.camera_arbitrated_pairs
					<< ", owners " << ownership_plan.stats.left_owned_pairs << "/"
					<< ownership_plan.stats.right_owned_pairs
					<< ", unmatched units " << ownership_plan.stats.unmatched_units << std::endl;
				if (!route_ownership)
					route_error = "no mutually matched hierarchy units";
			}
			if (!route_ownership)
				std::cerr << "Warning: route ownership disabled; using center-distance fallback: "
					<< route_error << std::endl;
		}
		else
		{
			std::cerr << "Warning: route ownership currently requires exactly two chunks; "
				<< "using center-distance fallback" << std::endl;
		}
		// Spatial ownership is active even when no reliable duplicate pair was found.
		std::vector<Gaussian> gaussians; 
		ExplicitTreeNode* root = new ExplicitTreeNode;

		for (int chunk_id(0); chunk_id < chunk_count; chunk_id++)
		{
			int argidx(chunk_id + 5);
			std::cout << "Adding hierarchy for chunk " << argv[argidx] << std::endl;
			const std::string& hierpath = hierarchy_paths[chunk_id];
			std::cout << "Hierarchy file path: " << hierpath << std::endl;
			
			ExplicitTreeNode* chunkRoot = new ExplicitTreeNode;
			const bool contributes = HierarchyExplicitLoader::loadExplicit(
				hierpath.c_str(), gaussians, chunkRoot, chunk_id, chunk_centers,
				spatial_ownership ? &spatial_plan.excluded_nodes[chunk_id]
					: (route_ownership ? &ownership_plan.excluded_nodes[chunk_id] : nullptr),
				route_ownership, spatial_ownership);
			if (!contributes)
			{
				delete chunkRoot;
				std::cout << "Chunk contributes no geometry: all units reliably duplicated" << std::endl;
				continue;
			}

			if (root->children.empty())
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
			root->merged.push_back(chunkRoot->merged.empty()
				? gaussians.at(chunkRoot->leaf_indices.at(0)) : chunkRoot->merged[0]);
			root->bounds.maxx[3] = 1e9f;
			root->bounds.minn[3] = 1e9f;
		}
		if (root->children.empty() || gaussians.empty())
			throw std::runtime_error("Merged scene contains no Gaussian leaves");
		if (root->children.size() > 1) {
			Gaussian gaussian = AvgMerger::mergeGaussians(root->merged);
			root->merged.clear();
			root->merged.emplace_back(gaussian);
		}

		if (!write_ply) {
			Writer::writeHierarchy(
				outputpath.c_str(),
				gaussians, root, true);
		}
		else {
			gaussians.insert(gaussians.begin(), scaffold_gaussians.begin(), scaffold_gaussians.end());
			Writer::writePly(outputpath.c_str(), gaussians, sh_degree);
		}
	}
}