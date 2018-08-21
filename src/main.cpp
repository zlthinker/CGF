#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <utility>

//#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot.h>
#include <pcl/features/shot_lrf.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/shot_lrf_omp.h>
#include <pcl/features/board.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>

#include "liblzf-3.6/lzf_c.c"
#include "liblzf-3.6/lzf_d.c"

using namespace std;

void usage(const char* program)
{
    cout << "Usage: " << program << " [options] <input.pcd>" << endl << endl;
    cout << "Options: " << endl;
    cout << "--relative If selected, scale is relative to the diameter of the model (-d). Otherwise scale is absolute." << endl;
    cout << "-r R Number of subdivisions in the radial direction. Default 17." << endl;
    cout << "-p P Number of subdivisions in the elevation direction. Default 11." << endl;
    cout << "-a A Number of subdivisions in the azimuth direction. Default 12." << endl;
    cout << "-s S Radius of sphere around each point. Default 1.18 (absolute) or 17\% of diameter (relative)." << endl;
    cout << "-d D Diameter of full model. Must be provided for relative scale." << endl;
    cout << "-m M Smallest radial subdivision. Default 0.1 (absolute) or 1.5\% of diameter (relative)." << endl;
    cout << "-l L Search radius for local reference frame. Default 0.25 (absolute) or 2\% of diameter (relative)." << endl;
    cout << "-t T Number of threads. Default 16." << endl;
    cout << "-o Output file name." << endl;
    cout << "-h Help menu." << endl;
}

vector<vector<double> > compute_intensities(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                                            pcl::PointCloud<pcl::PointNormal>::Ptr normals,
                                            pcl::PointCloud<pcl::PointXYZ>::Ptr keypoint_cloud,
                                            int num_bins_radius, 
                                            int num_bins_polar,
                                            int num_bins_azimuth,
                                            double search_radius,
                                            double lrf_radius, 
                                            double rmin,
                                            int num_threads)
{
    vector<vector<double> > intensities;
    intensities.resize(keypoint_cloud->points.size());
    
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud);

    pcl::PointCloud<pcl::ReferenceFrame>::Ptr frames(new pcl::PointCloud<pcl::ReferenceFrame>());
    pcl::SHOTLocalReferenceFrameEstimationOMP<pcl::PointXYZ>::Ptr lrf_estimator(new pcl::SHOTLocalReferenceFrameEstimationOMP<pcl::PointXYZ>());
    lrf_estimator->setRadiusSearch(lrf_radius);
    lrf_estimator->setSearchSurface(cloud);
    lrf_estimator->setInputCloud(keypoint_cloud);
    
    lrf_estimator->compute(*frames);

    pcl::StopWatch watch_intensities;

    double ln_rmin = log(rmin);
    double ln_rmax_rmin = log(search_radius/rmin);
    
    double azimuth_interval = 360.0 / num_bins_azimuth;
    double polar_interval = 180.0 / num_bins_polar; 
    vector<double> radii_interval, azimuth_division, polar_division;
    for(int i = 0; i < num_bins_radius+1; i++) {
        radii_interval.push_back(exp(ln_rmin + ((double)i) / num_bins_radius * ln_rmax_rmin));
    }
    for(int i = 0; i < num_bins_azimuth + 1; i++) {
        azimuth_division.push_back(i * azimuth_interval);
    } 
    for(int i = 0; i < num_bins_polar + 1; i++) {
        polar_division.push_back(i * polar_interval);
    } 
    radii_interval[0] = 0;

    vector<double> integr_radius, integr_polar;
    double integr_azimuth;
    for(int i = 0; i < num_bins_radius; i++) {
        integr_radius.push_back((radii_interval[i+1]*radii_interval[i+1]*radii_interval[i+1])/3 - (radii_interval[i]*radii_interval[i]*radii_interval[i])/3 );
    }
    integr_azimuth = pcl::deg2rad(azimuth_division[1]) - pcl::deg2rad(azimuth_division[0]);
    for(int i = 0; i < num_bins_polar; i++) {
        integr_polar.push_back(cos(pcl::deg2rad(polar_division[i]))-cos(pcl::deg2rad(polar_division[i+1])));
    }  

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for(int i = 0; i < keypoint_cloud->points.size(); i++) {
        vector<int> indices;
        vector<float> distances;
        vector<double> intensity;
        int sum = 0;
        intensity.resize(num_bins_radius * num_bins_polar * num_bins_azimuth);
        fill(intensity.begin(), intensity.end(), 0);

        tree->radiusSearch(keypoint_cloud->points[i], search_radius, indices, distances);

	if (indices.size() == 0)
	{
		intensities[i] = vector<double>(num_bins_radius * num_bins_polar * num_bins_azimuth, 0.0);
		continue;
	}

        int indice = indices[0];
        //std::cout << i << ", indice: " << indice << "\t, dist: " << distances[0] << "\n";
        //std::cout << "# indices: " << indices.size() << "\n";

        pcl::ReferenceFrame current_frame = (*frames)[i];
        Eigen::Vector4f current_frame_x (current_frame.x_axis[0], current_frame.x_axis[1], current_frame.x_axis[2], 0);
        Eigen::Vector4f current_frame_y (current_frame.y_axis[0], current_frame.y_axis[1], current_frame.y_axis[2], 0);
        Eigen::Vector4f current_frame_z (current_frame.z_axis[0], current_frame.z_axis[1], current_frame.z_axis[2], 0);

        if(isnan(current_frame_x[0]) || isnan(current_frame_x[1]) || isnan(current_frame_x[2]) ) {
            current_frame_x[0] = 1, current_frame_x[1] = 0, current_frame_x[2] = 0;  
            current_frame_y[0] = 0, current_frame_y[1] = 1, current_frame_y[2] = 0;  
            current_frame_z[0] = 0, current_frame_z[1] = 0, current_frame_z[2] = 1;  
        } else {
            float nx = normals->points[indice].normal_x, ny = normals->points[indice].normal_y, nz = normals->points[indice].normal_z;
            Eigen::Vector4f n(nx, ny, nz, 0);
            if(current_frame_z.dot(n) < 0) {
                current_frame_x = -current_frame_x;
                current_frame_y = -current_frame_y;
                current_frame_z = -current_frame_z;
            }
        }

        for(int j = 1; j < indices.size(); j++) {
            if(distances[j] > 1E-15) {
                Eigen::Vector4f v = cloud->points[indices[j]].getVector4fMap() - cloud->points[indice].getVector4fMap();
                double x_l = (double)v.dot(current_frame_x);
                double y_l = (double)v.dot(current_frame_y);
                double z_l = (double)v.dot(current_frame_z);
                
                double r = sqrt(x_l*x_l + y_l*y_l + z_l*z_l);
		if (r < 1E-8) { r = 1; }
                double theta = pcl::rad2deg(acos(z_l / r));
                double phi = pcl::rad2deg(atan2(y_l, x_l));
                int bin_r = int((num_bins_radius - 1) * (log(r) - ln_rmin) / ln_rmax_rmin + 1);
                int bin_theta = int(num_bins_polar * theta / 180);
                int bin_phi = int(num_bins_azimuth * (phi + 180) / 360);

                bin_r = bin_r >= 0 ? bin_r : 0;
                bin_r = bin_r < num_bins_radius ? bin_r : num_bins_radius - 1;
                bin_theta = bin_theta < num_bins_polar ? bin_theta : num_bins_polar - 1;
                bin_phi = bin_phi < num_bins_azimuth ? bin_phi : num_bins_azimuth - 1;
                int idx = bin_r + bin_theta * num_bins_radius + bin_phi * num_bins_radius * num_bins_polar;
                intensity[idx] += 1;
                sum += 1;
            }
        }
        if(sum > 0) {
            for(int j = 0; j < intensity.size(); j++) {
                intensity[j] /= sum;
            }
        }
        intensities[i] = intensity;
    }
    pcl::console::print_highlight("Raw Spherical Histograms Time: %f (s)\n", watch_intensities.getTimeSeconds());
    return intensities;
}

void ReadKeypointCloud(std::string const & keypoint_file,
                       pcl::PointCloud<pcl::PointXYZ>::Ptr & keypoint_cloud)
{
    std::ifstream fin(keypoint_file);
    size_t kpt_num;
    if (fin.is_open())
    {
        fin >> kpt_num;
        for (int i = 0; i < kpt_num; i++)
        {
            float x, y, z, scale, angle, response, octave;
            fin >> x >> y >> z >> scale >> angle >> response >> octave;
            pcl::PointXYZ point(x, y, z);
            keypoint_cloud->push_back(point);
        }
    }
}

int main(int argc, char* argv[])
{
    int num_bins_radius = 17, num_bins_polar = 11, num_bins_azimuth = 12;
    int num_threads = 16;
    double search_radius = 1.18, lrf_radius = 0.25;
    double diameter = 4*sqrt(3);
    double rmin = 0.1;
    string output_file;
    string keypoint_file;
 
    if(pcl::console::find_argument(argc, argv, "-h") >= 0 || argc == 1) {
        usage(argv[0]);
        return 0;
    }
    
    pcl::console::parse_argument(argc, argv, "-r", num_bins_radius);
    pcl::console::parse_argument(argc, argv, "-p", num_bins_polar);
    pcl::console::parse_argument(argc, argv, "-a", num_bins_azimuth);
    pcl::console::parse_argument(argc, argv, "-s", search_radius);
    pcl::console::parse_argument(argc, argv, "-m", rmin);
    pcl::console::parse_argument(argc, argv, "-l", lrf_radius);
    pcl::console::parse_argument(argc, argv, "-o", output_file);
    pcl::console::parse_argument(argc, argv, "-t", num_threads);
    pcl::console::parse_argument(argc, argv, "-d", diameter);
    pcl::console::parse_argument(argc, argv, "-kpt", keypoint_file);

    bool relative_scale = pcl::console::find_argument(argc, argv, "--relative") >= 0;
    std::cout << relative_scale << std::endl;
    if(relative_scale) {
        search_radius = 0.17 * diameter;
        lrf_radius = 0.02 * diameter;
        rmin = 0.015 * diameter; 
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);
     pcl::PointCloud<pcl::PointXYZ>::Ptr keypoint_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    ofstream fout(output_file.c_str(), ios::binary);
    vector<double> intensities_flat;

    string pcd_name = argv[argc-1]; 
    int success = pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_name, *cloud);
    if(success == -1) {
        PCL_ERROR("Could not read file.");
        return 1;
    }
    success = pcl::io::loadPCDFile<pcl::PointNormal>(pcd_name, *cloud_with_normals);
    if(success == -1) {
        PCL_ERROR("Could not read file.");
        return 1;
    }
    std::cout << "Read keypoint file: " << keypoint_file << "\n";
    ReadKeypointCloud(keypoint_file, keypoint_cloud);
    std::cout << "Read keypoint file finished.\n";

    vector<vector<double> > intensities = compute_intensities(cloud, cloud_with_normals, keypoint_cloud,
                                                              num_bins_radius, num_bins_polar, num_bins_azimuth, 
                                                              search_radius, lrf_radius, 
                                                              rmin, num_threads);
    std::cout << "Compute intensity finished.\n";
    std::cout << "# intensities: " << intensities.size() << "\n";
    for(int i = 0; i < intensities.size(); i++) {
        for(int j = 0; j < intensities[i].size(); j++) {
            intensities_flat.push_back(intensities[i][j]);
        }
    }

    if(intensities_flat.size()*sizeof(double) > 4294967293){
        std::cout << "Warning: More than 4294967293 bytes allocated. Compression may not work as expected." << std::endl;
    }
        
    vector<unsigned char> intensities_bytes;
    intensities_bytes.resize(intensities_flat.size()*sizeof(double));
    std::cout << "# bytes: " << intensities_bytes.size() << "\n";
    memcpy(&intensities_bytes[0], &intensities_flat[0], intensities_flat.size()*sizeof(double));
    vector<unsigned char> intensities_compressed;
    intensities_compressed.resize(intensities_bytes.size()); 
    std::cout << "Before lzf compress.\n";
    size_t compressed_len = lzf_compress(&intensities_bytes[0], intensities_bytes.size(), &intensities_compressed[0], intensities_bytes.size());
    std::cout << "After lzf compress.\n";
    fout.write(reinterpret_cast<const char*>(&intensities_compressed[0]), compressed_len);
    fout.close();
    return 0;
}
