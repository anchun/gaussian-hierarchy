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
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

namespace
{
uint32_t finishPlyLoad(std::vector<Gaussian>& gaussians, uint32_t sh_degree)
{
	auto invalid_begin = std::remove_if(gaussians.begin(), gaussians.end(), [](const Gaussian& g) {
		return !g.position.allFinite() || !g.shs.allFinite() || !std::isfinite(g.opacity) ||
			!g.scale.allFinite() || (g.scale.array() <= 0).any() ||
			!g.rotation.allFinite() || !g.covariance.allFinite();
	});
	const size_t invalid_count = std::distance(invalid_begin, gaussians.end());
	gaussians.erase(invalid_begin, gaussians.end());
	if (invalid_count > 0)
		std::cout << "Skipping " << invalid_count << " non-finite Gaussian(s) from PLY" << std::endl;
	if (gaussians.empty())
		throw std::runtime_error("PLY contains no valid Gaussians!");
	return sh_degree;
}
}

uint32_t Loader::loadPlyDir(const char* filename, std::vector<Gaussian>& gaussians)
{
	int num_skip;
	std::ifstream cfgfile(std::string(filename) + "/pc_info.txt");
	cfgfile >> num_skip;
	std::cout << "Skipping " << num_skip << std::endl;

	std::ifstream infile(std::string(filename) + "/point_cloud.ply", std::ios_base::binary);

	if (!infile.good())
		throw std::runtime_error("File not found!");

	std::string buff;
	std::getline(infile, buff);
	std::getline(infile, buff);

	std::string dummy;
	std::getline(infile, buff);
	std::stringstream ss(buff);
	int count;
	ss >> dummy >> dummy >> count;

	while (std::getline(infile, buff))
		if (buff.compare("end_header") == 0)
			break;

	gaussians.resize(count);

	std::vector<RichPointDegree1> points(count);
	infile.read((char*)points.data(), count * sizeof(RichPointDegree1));

	for (int i = 0; i < gaussians.size() - num_skip; i++)
	{
		Gaussian& g = gaussians[i];
		RichPointDegree1& p = points[i + num_skip];

		g.opacity = sigmoid(p.opacity);
		g.position = p.position;
		g.rotation = Eigen::Vector4f(p.rotation[0], p.rotation[1], p.rotation[2], p.rotation[3]).normalized();
		g.scale = p.scale.array().exp();
		for (int j = 0; j < 3; j++)
			g.shs[j] = p.shs[j];
		for (int j = 1; j < 4; j++)
		{
			g.shs[(j - 1) + 3]  = p.shs[j * 3 + 0];
			g.shs[(j - 1) + 18] = p.shs[j * 3 + 1];
			g.shs[(j - 1) + 33] = p.shs[j * 3 + 2];
		}
		for (int j = 4; j < 16; j++)
		{
			g.shs[(j - 1) + 3]  = 0;
			g.shs[(j - 1) + 18] = 0;
			g.shs[(j - 1) + 33] = 0;
		}
		computeCovariance(g.scale, g.rotation, g.covariance);
	}
	return 1; // sh_degree
}

uint32_t Loader::loadPly(const char* filename, std::vector<Gaussian>& gaussians, int skyboxpoints)
{
	std::ifstream infile(filename, std::ios_base::binary);
	std::cout << filename << std::endl;
 	if (!infile.good())
		throw std::runtime_error("File not found!");

	std::string buff;
	int count = 0;
	while (true)
	{
		std::getline(infile, buff);
		std::stringstream ss(buff);
		std::string dummy;
		ss >> dummy;
		if (dummy == "element") {
			ss  >> dummy >> count;
			break;
		}
	}

	int property_count = 0;
	std::vector<std::string> property_names;
	while (std::getline(infile, buff))
	{
		if (buff.compare("end_header") == 0)
			break;
		std::stringstream ss(buff);
		std::string keyword;
		ss >> keyword;
		if (keyword == "property") {
			std::string property_type;
			std::string property_name;
			ss >> property_type >> property_name;
			property_count++;
			property_names.push_back(property_name);
		}
	}

	if (count <= skyboxpoints)
		throw std::runtime_error("PLY contains no Gaussians after skipping skybox points!");

	const std::vector<std::string> reordered_degree0_properties = {
		"x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3",
		"scale_0", "scale_1", "scale_2", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"
	};

	if (property_names == reordered_degree0_properties) {
		std::vector<std::array<float, 14>> points(count);
		infile.read((char*)points.data(), count * sizeof(points[0]));
		if (!infile)
			throw std::runtime_error("Unexpected end of PLY vertex data!");

		gaussians.resize(count - skyboxpoints);
		for (int i = 0; i < gaussians.size(); i++)
		{
			Gaussian& g = gaussians[i];
			const auto& p = points[skyboxpoints + i];

			g.position = Eigen::Vector3f(p[0], p[1], p[2]);
			g.rotation = Eigen::Vector4f(p[3], p[4], p[5], p[6]).normalized();
			g.scale = Eigen::Vector3f(p[7], p[8], p[9]).array().exp();
			g.opacity = sigmoid(p[10]);
			g.shs.setZero();
			g.shs[0] = p[11];
			g.shs[1] = p[12];
			g.shs[2] = p[13];
			computeCovariance(g.scale, g.rotation, g.covariance);
		}
		return finishPlyLoad(gaussians, 0);
	}
	else if (property_count == sizeof(RichPoint) / sizeof(float)) {
		gaussians.resize(count - skyboxpoints);
		std::vector<RichPoint> points(count);
		infile.read((char*)points.data(), count * sizeof(RichPoint));

		for (int i = 0; i < gaussians.size(); i++)
		{
			Gaussian& g = gaussians[i];
			RichPoint& p = points[skyboxpoints + i];

			g.opacity = sigmoid(p.opacity);
			g.position = p.position;
			g.rotation = Eigen::Vector4f(p.rotation[0], p.rotation[1], p.rotation[2], p.rotation[3]).normalized();
			g.scale = p.scale.array().exp();
			g.shs.setZero();
			for (int j = 0; j < 3; j++)
				g.shs[j] = p.shs[j];
			for (int j = 1; j < 16; j++)
			{
				g.shs[j * 3 + 0] = p.shs[(j - 1) + 3];
				g.shs[j * 3 + 1] = p.shs[(j - 1) + 18];
				g.shs[j * 3 + 2] = p.shs[(j - 1) + 33];
			}
			computeCovariance(g.scale, g.rotation, g.covariance);
		}
		return finishPlyLoad(gaussians, 3); //sh_degree
	}
	else if (property_count == sizeof(RichPointDegree1WithNormal) / sizeof(float)) {
		gaussians.resize(count - skyboxpoints);
		std::vector<RichPointDegree1WithNormal> points(count);
		infile.read((char*)points.data(), count * sizeof(RichPointDegree1WithNormal));

		for (int i = 0; i < gaussians.size(); i++)
		{
			Gaussian& g = gaussians[i];
			RichPointDegree1WithNormal& p = points[skyboxpoints + i];

			g.opacity = sigmoid(p.opacity);
			g.position = p.position;
			g.rotation = Eigen::Vector4f(p.rotation[0], p.rotation[1], p.rotation[2], p.rotation[3]).normalized();
			g.scale = p.scale.array().exp();
			g.shs.setZero();
			for (int j = 0; j < 3; j++)
				g.shs[j] = p.shs[j];
			for (int j = 1; j < 4; j++)
			{
				g.shs[j * 3 + 0] = p.shs[(j - 1) + 3];
				g.shs[j * 3 + 1] = p.shs[(j - 1) + 6];
				g.shs[j * 3 + 2] = p.shs[(j - 1) + 9];
			}
			computeCovariance(g.scale, g.rotation, g.covariance);
		}
		return finishPlyLoad(gaussians, 1); //sh_degree
	}
	else if (property_count == sizeof(RichPointDegree1) / sizeof(float)) {
		gaussians.resize(count - skyboxpoints);
		std::vector<RichPointDegree1> points(count);
		infile.read((char*)points.data(), count * sizeof(RichPointDegree1));

		for (int i = 0; i < gaussians.size(); i++)
		{
			Gaussian& g = gaussians[i];
			RichPointDegree1& p = points[skyboxpoints + i];

			g.opacity = sigmoid(p.opacity);
			g.position = p.position;
			g.rotation = Eigen::Vector4f(p.rotation[0], p.rotation[1], p.rotation[2], p.rotation[3]).normalized();
			g.scale = p.scale.array().exp();
			g.shs.setZero();
			for (int j = 0; j < 3; j++)
				g.shs[j] = p.shs[j];
			for (int j = 1; j < 4; j++)
			{
				g.shs[j * 3 + 0] = p.shs[(j - 1) + 3];
				g.shs[j * 3 + 1] = p.shs[(j - 1) + 6];
				g.shs[j * 3 + 2] = p.shs[(j - 1) + 9];
			}
			computeCovariance(g.scale, g.rotation, g.covariance);
		}
		return finishPlyLoad(gaussians, 1); //sh_degree
	}
	else if (property_count == sizeof(RichPointDegree0WithNormal) / sizeof(float)) {
		gaussians.resize(count - skyboxpoints);
		std::vector<RichPointDegree0WithNormal> points(count);
		infile.read((char*)points.data(), count * sizeof(RichPointDegree0WithNormal));

		for (int i = 0; i < gaussians.size(); i++)
		{
			Gaussian& g = gaussians[i];
			RichPointDegree0WithNormal& p = points[skyboxpoints + i];

			g.opacity = sigmoid(p.opacity);
			g.position = p.position;
			g.rotation = Eigen::Vector4f(p.rotation[0], p.rotation[1], p.rotation[2], p.rotation[3]).normalized();
			g.scale = p.scale.array().exp();
			g.shs.setZero();
			for (int j = 0; j < 3; j++)
				g.shs[j] = p.shs[j];
			computeCovariance(g.scale, g.rotation, g.covariance);
		}
		return finishPlyLoad(gaussians, 0); //sh_degree
	}
	else if (property_count == sizeof(RichPointDegree0) / sizeof(float)) {
		gaussians.resize(count - skyboxpoints);
		std::vector<RichPointDegree0> points(count);
		infile.read((char*)points.data(), count * sizeof(RichPointDegree0));

		for (int i = 0; i < gaussians.size(); i++)
		{
			Gaussian& g = gaussians[i];
			RichPointDegree0& p = points[skyboxpoints + i];

			g.opacity = sigmoid(p.opacity);
			g.position = p.position;
			g.rotation = Eigen::Vector4f(p.rotation[0], p.rotation[1], p.rotation[2], p.rotation[3]).normalized();
			g.scale = p.scale.array().exp();
			g.shs.setZero();
			for (int j = 0; j < 3; j++)
				g.shs[j] = p.shs[j];
			computeCovariance(g.scale, g.rotation, g.covariance);
		}
		return finishPlyLoad(gaussians, 0); //sh_degree
	}
	else {
		std::cout << "Invalid ply files with property_count" << property_count << std::endl;
		throw std::runtime_error("Invalid ply files!");
	}
	return 0; //sh_degree
}

uint32_t Loader::loadBin(const char* filename, std::vector<Gaussian>& gaussians, int skyboxpoints)
{
	std::ifstream infile(filename, std::ios_base::binary);

	std::cout << filename << std::endl;
	if (!infile.good())
		throw std::runtime_error("File not found!");

	int count;
	infile.read((char*)&count, sizeof(int));

	std::vector<Eigen::Vector3f> pos(count);
	std::vector<SHs> shs(count);
	std::vector<Eigen::Vector3f> scales(count);
	std::vector<Eigen::Vector4f> rot(count);
	std::vector<float> alphas(count);

	gaussians.resize(count - skyboxpoints);

	infile.read((char*)pos.data(), sizeof(float) * 3 * count);
	infile.read((char*)shs.data(), sizeof(SHs) * count);
	infile.read((char*)alphas.data(), sizeof(float) * count);
	infile.read((char*)scales.data(), sizeof(float) * 3 * count);
	infile.read((char*)rot.data(), sizeof(float) * 4 * count);

	for (int i = 0; i < gaussians.size(); i++)
	{
		int k = i + skyboxpoints;
		Gaussian& g = gaussians[i];

		g.opacity = sigmoid(alphas[k]);
		g.position = pos[k];
		g.rotation = Eigen::Vector4f(rot[k][0], rot[k][1], rot[k][2], rot[k][3]).normalized();
		g.scale = scales[k].array().exp();
		g.shs = shs[k];
		computeCovariance(g.scale, g.rotation, g.covariance);
	}
	return 3; //sh_degree
}