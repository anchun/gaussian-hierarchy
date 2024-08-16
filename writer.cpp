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


#include "writer.h"
#include <iostream>
#include <fstream>
#include "hierarchy_writer.h"
#include <map>

void populateRec(
	const ExplicitTreeNode* treenode,
	int id,
	const std::vector<Gaussian>& gaussians, 
	std::vector<Eigen::Vector3f>& positions,
	std::vector<Eigen::Vector4f>& rotations,
	std::vector<Eigen::Vector3f>& log_scales,
	std::vector<float>& opacities,
	std::vector<SHs>& shs,
	std::vector<Node>& basenodes,
	std::vector<Box>& boxes,
	std::map<int, const ExplicitTreeNode*>* base2tree = nullptr)
{
	if(base2tree)
		base2tree->insert(std::make_pair(id, treenode));

	boxes[id] = treenode->bounds;
	basenodes[id].start = positions.size();
	for (auto& i : treenode->leaf_indices)
	{
		const Gaussian& g = gaussians[i];
		positions.push_back(g.position);
		rotations.push_back(g.rotation);
		log_scales.push_back(g.scale.array().log());
		opacities.push_back(g.opacity);
		shs.push_back(g.shs);
	}
	basenodes[id].count_leafs = treenode->leaf_indices.size();

	for (auto& g : treenode->merged)
	{
		positions.push_back(g.position);
		rotations.push_back(g.rotation);
		log_scales.push_back(g.scale.array().log());
		opacities.push_back(g.opacity);
		shs.push_back(g.shs);
	}
	basenodes[id].count_merged = treenode->merged.size();

	basenodes[id].start_children = basenodes.size();
	for (int n = 0; n < treenode->children.size(); n++)
	{
		basenodes.push_back(Node());
		basenodes.back().parent = id;
		boxes.push_back(Box());
	}
	basenodes[id].count_children = treenode->children.size();

	basenodes[id].depth = treenode->depth;

	for (int n = 0; n < treenode->children.size(); n++)
	{
		populateRec(
			treenode->children[n],
			basenodes[id].start_children + n,
			gaussians, 
			positions, 
			rotations, 
			log_scales, 
			opacities,
			shs, 
			basenodes,
			boxes,
			base2tree);
	}
}

void recTraverse(int id, std::vector<Node>& nodes, int& count)
{
	if (nodes[id].depth == 0)
		count++;
	if (nodes[id].count_children != 0 && nodes[id].depth == 0)
		throw std::runtime_error("An error occurred in traversal");
	for (int i = 0; i < nodes[id].count_children; i++)
	{
		recTraverse(nodes[id].start_children + i, nodes, count);
	}
}

void Writer::makeHierarchy(
	const std::vector<Gaussian>& gaussians,
	const ExplicitTreeNode* root,
	std::vector<Eigen::Vector3f>& positions,
	std::vector<Eigen::Vector4f>& rotations,
	std::vector<Eigen::Vector3f>& log_scales,
	std::vector<float>& opacities,
	std::vector<SHs>& shs,
	std::vector<Node>& basenodes,
	std::vector<Box>& boxes,
	std::map<int, const ExplicitTreeNode*>* base2tree)
{
	basenodes.resize(1);
	boxes.resize(1);

	populateRec(
		root,
		0,
		gaussians,
		positions, rotations, log_scales, opacities, shs, basenodes, boxes,
		base2tree);
}

void Writer::writeHierarchy(const char* filename, const std::vector<Gaussian>& gaussians, const ExplicitTreeNode* root, bool compressed)
{
	std::vector<Eigen::Vector3f> positions;
	std::vector<Eigen::Vector4f> rotations;
	std::vector<Eigen::Vector3f> log_scales;
	std::vector<float> opacities;
	std::vector<SHs> shs;
	std::vector<Node> basenodes;
	std::vector<Box> boxes;

	makeHierarchy(gaussians, root, positions, rotations, log_scales, opacities, shs, basenodes, boxes);

	HierarchyWriter writer;
	writer.write(
		filename,
		positions.size(),
		basenodes.size(),
		positions.data(),
		shs.data(),
		opacities.data(),
		log_scales.data(),
		rotations.data(),
		basenodes.data(),
		boxes.data(),
		compressed
	);
}

void writePlyDegree3(const char* filename, const std::vector<Gaussian>& gaussians)
{
	size_t gaussianCount = gaussians.size();
	// data prepare
	std::vector<RichPoint> points(gaussianCount);
	for (size_t i = 0; i < gaussianCount; i++)
	{
		const Gaussian& g = gaussians[i];
		RichPoint& p = points[i];
		p.position = g.position;
		p.normal = Eigen::Vector3f(0, 0, 0);
		for (int j = 0; j < 3; j++)
			p.shs[j] = g.shs[j];
		for (int j = 1; j < 16; j++)
		{
			p.shs[(j - 1) + 3] = g.shs[j * 3 + 0];
			p.shs[(j - 1) + 18] = g.shs[j * 3 + 1];
			p.shs[(j - 1) + 33] = g.shs[j * 3 + 2];
		}
		double opacity = std::clamp((double)g.opacity, 1e-12, 1.0 - 1e-12);
		p.opacity = (float)log(opacity / (1 - opacity));
		p.scale = g.scale.array().log();
		p.rotation[0] = g.rotation[0];
		p.rotation[1] = g.rotation[1];
		p.rotation[2] = g.rotation[2];
		p.rotation[3] = g.rotation[3];
	}
	
	std::ofstream outfile(filename, std::ios_base::binary);
	if (!outfile.good())
		throw std::runtime_error("File not created!");

	outfile << "ply" << std::endl;
	outfile << "format binary_little_endian 1.0" << std::endl;
	outfile << "element vertex " << points.size() << std::endl;
	outfile << "property float x" << std::endl;
	outfile << "property float y" << std::endl;
	outfile << "property float z" << std::endl;
	outfile << "property float nx" << std::endl;
	outfile << "property float ny" << std::endl;
	outfile << "property float nz" << std::endl;
	outfile << "property float f_dc_0" << std::endl;
	outfile << "property float f_dc_1" << std::endl;
	outfile << "property float f_dc_2" << std::endl;
	outfile << "property float f_rest_0" << std::endl;
	outfile << "property float f_rest_1" << std::endl;
	outfile << "property float f_rest_2" << std::endl;
	outfile << "property float f_rest_3" << std::endl;
	outfile << "property float f_rest_4" << std::endl;
	outfile << "property float f_rest_5" << std::endl;
	outfile << "property float f_rest_6" << std::endl;
	outfile << "property float f_rest_7" << std::endl;
	outfile << "property float f_rest_8" << std::endl;
	outfile << "property float f_rest_9" << std::endl;
	outfile << "property float f_rest_10" << std::endl;
	outfile << "property float f_rest_11" << std::endl;
	outfile << "property float f_rest_12" << std::endl;
	outfile << "property float f_rest_13" << std::endl;
	outfile << "property float f_rest_14" << std::endl;
	outfile << "property float f_rest_15" << std::endl;
	outfile << "property float f_rest_16" << std::endl;
	outfile << "property float f_rest_17" << std::endl;
	outfile << "property float f_rest_18" << std::endl;
	outfile << "property float f_rest_19" << std::endl;
	outfile << "property float f_rest_20" << std::endl;
	outfile << "property float f_rest_21" << std::endl;
	outfile << "property float f_rest_22" << std::endl;
	outfile << "property float f_rest_23" << std::endl;
	outfile << "property float f_rest_24" << std::endl;
	outfile << "property float f_rest_25" << std::endl;
	outfile << "property float f_rest_26" << std::endl;
	outfile << "property float f_rest_27" << std::endl;
	outfile << "property float f_rest_28" << std::endl;
	outfile << "property float f_rest_29" << std::endl;
	outfile << "property float f_rest_30" << std::endl;
	outfile << "property float f_rest_31" << std::endl;
	outfile << "property float f_rest_32" << std::endl;
	outfile << "property float f_rest_33" << std::endl;
	outfile << "property float f_rest_34" << std::endl;
	outfile << "property float f_rest_35" << std::endl;
	outfile << "property float f_rest_36" << std::endl;
	outfile << "property float f_rest_37" << std::endl;
	outfile << "property float f_rest_38" << std::endl;
	outfile << "property float f_rest_39" << std::endl;
	outfile << "property float f_rest_40" << std::endl;
	outfile << "property float f_rest_41" << std::endl;
	outfile << "property float f_rest_42" << std::endl;
	outfile << "property float f_rest_43" << std::endl;
	outfile << "property float f_rest_44" << std::endl;
	outfile << "property float opacity" << std::endl;
	outfile << "property float scale_0" << std::endl;
	outfile << "property float scale_1" << std::endl;
	outfile << "property float scale_2" << std::endl;
	outfile << "property float rot_0" << std::endl;
	outfile << "property float rot_1" << std::endl;
	outfile << "property float rot_2" << std::endl;
	outfile << "property float rot_3" << std::endl;
	outfile << "end_header" << std::endl;
	outfile.write((char*)points.data(), points.size() * sizeof(RichPoint));
	outfile.close();
	std::cout << "writing succeed: " << filename << std::endl;
}

void writePlyDegree1(const char* filename, const std::vector<Gaussian>& gaussians)
{
	size_t gaussianCount = gaussians.size();
	// data prepare
	std::vector<RichPointDegree1> points(gaussianCount);
	for (size_t i = 0; i < gaussianCount; i++)
	{
		const Gaussian& g = gaussians[i];
		RichPointDegree1& p = points[i];
		p.position = g.position;
		p.normal = Eigen::Vector3f(0, 0, 0);
		for (int j = 0; j < 3; j++)
			p.shs[j] = g.shs[j];
		for (int j = 1; j < 4; j++)
		{
			p.shs[(j - 1) + 3] = g.shs[j * 3 + 0];
			p.shs[(j - 1) + 6] = g.shs[j * 3 + 1];
			p.shs[(j - 1) + 9] = g.shs[j * 3 + 2];
		}
		double opacity = std::clamp((double)g.opacity, 1e-12, 1.0 - 1e-12);
		p.opacity = (float)log(opacity / (1 - opacity));
		p.scale = g.scale.array().log();
		p.rotation[0] = g.rotation[0];
		p.rotation[1] = g.rotation[1];
		p.rotation[2] = g.rotation[2];
		p.rotation[3] = g.rotation[3];
	}

	std::ofstream outfile(filename, std::ios_base::binary);
	if (!outfile.good())
		throw std::runtime_error("File not created!");

	outfile << "ply" << std::endl;
	outfile << "format binary_little_endian 1.0" << std::endl;
	outfile << "element vertex " << points.size() << std::endl;
	outfile << "property float x" << std::endl;
	outfile << "property float y" << std::endl;
	outfile << "property float z" << std::endl;
	outfile << "property float nx" << std::endl;
	outfile << "property float ny" << std::endl;
	outfile << "property float nz" << std::endl;
	outfile << "property float f_dc_0" << std::endl;
	outfile << "property float f_dc_1" << std::endl;
	outfile << "property float f_dc_2" << std::endl;
	outfile << "property float f_rest_0" << std::endl;
	outfile << "property float f_rest_1" << std::endl;
	outfile << "property float f_rest_2" << std::endl;
	outfile << "property float f_rest_3" << std::endl;
	outfile << "property float f_rest_4" << std::endl;
	outfile << "property float f_rest_5" << std::endl;
	outfile << "property float f_rest_6" << std::endl;
	outfile << "property float f_rest_7" << std::endl;
	outfile << "property float f_rest_8" << std::endl;
	outfile << "property float opacity" << std::endl;
	outfile << "property float scale_0" << std::endl;
	outfile << "property float scale_1" << std::endl;
	outfile << "property float scale_2" << std::endl;
	outfile << "property float rot_0" << std::endl;
	outfile << "property float rot_1" << std::endl;
	outfile << "property float rot_2" << std::endl;
	outfile << "property float rot_3" << std::endl;
	outfile << "end_header" << std::endl;
	outfile.write((char*)points.data(), points.size() * sizeof(RichPointDegree1));
	outfile.close();
	std::cout << "writing succeed: " << filename << std::endl;
}

void writePlyDegree0(const char* filename, const std::vector<Gaussian>& gaussians)
{
	size_t gaussianCount = gaussians.size();
	// data prepare
	std::vector<RichPointDegree0> points(gaussianCount);
	for (size_t i = 0; i < gaussianCount; i++)
	{
		const Gaussian& g = gaussians[i];
		RichPointDegree0& p = points[i];
		p.position = g.position;
		for (int j = 0; j < 3; j++)
			p.shs[j] = g.shs[j];
		double opacity = std::clamp((double)g.opacity, 1e-12, 1.0 - 1e-12);
		p.opacity = (float)log(opacity / (1 - opacity));
		p.scale = g.scale.array().log();
		p.rotation[0] = g.rotation[0];
		p.rotation[1] = g.rotation[1];
		p.rotation[2] = g.rotation[2];
		p.rotation[3] = g.rotation[3];
	}

	std::ofstream outfile(filename, std::ios_base::binary);
	if (!outfile.good())
		throw std::runtime_error("File not created!");

	outfile << "ply" << std::endl;
	outfile << "format binary_little_endian 1.0" << std::endl;
	outfile << "element vertex " << points.size() << std::endl;
	outfile << "property float x" << std::endl;
	outfile << "property float y" << std::endl;
	outfile << "property float z" << std::endl;
	outfile << "property float f_dc_0" << std::endl;
	outfile << "property float f_dc_1" << std::endl;
	outfile << "property float f_dc_2" << std::endl;
	outfile << "property float opacity" << std::endl;
	outfile << "property float scale_0" << std::endl;
	outfile << "property float scale_1" << std::endl;
	outfile << "property float scale_2" << std::endl;
	outfile << "property float rot_0" << std::endl;
	outfile << "property float rot_1" << std::endl;
	outfile << "property float rot_2" << std::endl;
	outfile << "property float rot_3" << std::endl;
	outfile << "end_header" << std::endl;
	outfile.write((char*)points.data(), points.size() * sizeof(RichPointDegree0));
	outfile.close();
	std::cout << "writing succeed: " << filename << std::endl;
}

void Writer::writePly(const char* filename, const std::vector<Gaussian>& gaussians, std::uint32_t sh_degree)
{
	std::cout << "writing ply file with " << gaussians.size() << " gaussians in degree " << sh_degree << std::endl;
	if (sh_degree == 0) {
		writePlyDegree0(filename, gaussians);
	}
	else if (sh_degree == 1) {
		writePlyDegree1(filename, gaussians);
	}
	else if (sh_degree == 3) {
		writePlyDegree3(filename, gaussians);
	}
	else {
		std::cout << "not supported degrees:  " << sh_degree << std::endl;
	}
}
