#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include "line2Dup.h"

using namespace std;

// ROI control points enum
enum ControlPoint {
    NONE = -1,
    TL = 0,  // Top-left: move entire ROI
    TR = 1,  // Top-right: rotate
    L = 2,   // Left edge center: move left edge
    R = 3,   // Right edge center: move right edge
    T = 4,   // Top edge center: move top edge
    B = 5,   // Bottom edge center: move bottom edge
    CENTER = 6 // ROI center: for rotation reference
};

// ROI structure
struct ROI {
    cv::Point tl;  // Top-left corner
    cv::Point br;  // Bottom-right corner
    double angle;  // Rotation angle in degrees
    
    ROI() : tl(0, 0), br(100, 100), angle(0) {}
    ROI(cv::Point _tl, cv::Point _br) : tl(_tl), br(_br), angle(0) {}
    
    // Get ROI width
    int width() const { return br.x - tl.x; }
    
    // Get ROI height
    int height() const { return br.y - tl.y; }
    
    // Get ROI center
    cv::Point2f center() const {
        return cv::Point2f((tl.x + br.x) / 2.0f, (tl.y + br.y) / 2.0f);
    }
    
    // Get control points
    vector<cv::Point2f> getControlPoints() const {
        vector<cv::Point2f> points;
        
        // Top-left
        points.push_back(tl);
        // Top-right
        points.push_back(cv::Point2f(br.x, tl.y));
        // Left edge center
        points.push_back(cv::Point2f(tl.x, (tl.y + br.y) / 2.0f));
        // Right edge center
        points.push_back(cv::Point2f(br.x, (tl.y + br.y) / 2.0f));
        // Top edge center
        points.push_back(cv::Point2f((tl.x + br.x) / 2.0f, tl.y));
        // Bottom edge center
        points.push_back(cv::Point2f((tl.x + br.x) / 2.0f, br.y));
        // Center
        points.push_back(center());
        
        return points;
    }
    
    // Rotate a point around center by given angle
    cv::Point2f rotatePoint(const cv::Point2f &pt, const cv::Point2f &center, double angle) const {
        double rad = angle * M_PI / 180.0;
        double cosA = cos(rad);
        double sinA = sin(rad);
        
        cv::Point2f translated(pt.x - center.x, pt.y - center.y);
        cv::Point2f rotated(translated.x * cosA - translated.y * sinA, translated.x * sinA + translated.y * cosA);
        
        return cv::Point2f(rotated.x + center.x, rotated.y + center.y);
    }
    
    // Get rotated ROI vertices
    vector<cv::Point2f> getRotatedVertices() const {
        vector<cv::Point2f> vertices;
        cv::Point2f c = center();
        
        vertices.push_back(rotatePoint(tl, c, angle));
        vertices.push_back(rotatePoint(cv::Point2f(br.x, tl.y), c, angle));
        vertices.push_back(rotatePoint(br, c, angle));
        vertices.push_back(rotatePoint(cv::Point2f(tl.x, br.y), c, angle));
        
        return vertices;
    }
};

// Global variables
cv::Mat g_img;
cv::Mat g_img_copy;
ROI g_roi;
ControlPoint g_selected_point = NONE;
cv::Point g_mouse_pos;
bool g_dragging = false;

// Save ROI to file with template features
void saveROI(const ROI &roi, const string &filename) {
    // Extract ROI from image
    cv::Rect rect(roi.tl, roi.br);
    cv::Mat roi_img = g_img(rect);
    
    // Create mask for ROI
    cv::Mat mask(roi_img.size(), CV_8UC1, cv::Scalar(255));
    
    // Initialize ColorGradient with default parameters
    line2Dup::ColorGradient cg;
    
    // Create ColorGradientPyramid for feature extraction
    line2Dup::ColorGradientPyramid cgp(roi_img, mask, cg.weak_threshold, cg.num_features, cg.strong_threshold);
    
    // Extract template features
    line2Dup::Template templ;
    if (cgp.extractTemplate(templ)) {
        // Set template parameters
        templ.tl_x = roi.tl.x;
        templ.tl_y = roi.tl.y;
        templ.width = rect.width;
        templ.height = rect.height;
        
        // Save template to file
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        if (fs.isOpened()) {
            templ.write(fs);
            fs.release();
            cout << "Template saved to " << filename << endl;
        } else {
            cerr << "Failed to open file for writing template!" << endl;
        }
    } else {
        cerr << "Failed to extract template features!" << endl;
    }
}

// Draw ROI with control points
void drawROI(cv::Mat &img, const ROI &roi) {
    vector<cv::Point2f> vertices = roi.getRotatedVertices();
    
    // Draw ROI rectangle
    for (int i = 0; i < 4; ++i) {
        cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 255, 0), 2);
    }
    
    // Get control points
    vector<cv::Point2f> control_points = roi.getControlPoints();
    
    // Draw control points
    // Top-left (move) - red
    cv::circle(img, control_points[TL], 5, cv::Scalar(0, 0, 255), -1);
    // Top-right (rotate) - blue
    cv::circle(img, control_points[TR], 5, cv::Scalar(255, 0, 0), -1);
    // Left edge center - green
    cv::circle(img, control_points[L], 5, cv::Scalar(0, 255, 0), -1);
    // Right edge center - green
    cv::circle(img, control_points[R], 5, cv::Scalar(0, 255, 0), -1);
    // Top edge center - green
    cv::circle(img, control_points[T], 5, cv::Scalar(0, 255, 0), -1);
    // Bottom edge center - green
    cv::circle(img, control_points[B], 5, cv::Scalar(0, 255, 0), -1);
}

// Find which control point is clicked
ControlPoint findClickedPoint(const cv::Point &mouse_pt, const vector<cv::Point2f> &control_points, int threshold = 10) {
    cv::Point2f mouse_pt_f(mouse_pt);
    for (int i = 0; i < control_points.size(); ++i) {
        if (cv::norm(mouse_pt_f - control_points[i]) < threshold) {
            return static_cast<ControlPoint>(i);
        }
    }
    return NONE;
}

// Mouse callback function
void mouseCallback(int event, int x, int y, int flags, void *userdata) {
    g_mouse_pos = cv::Point(x, y);
    
    switch (event) {
        case cv::EVENT_LBUTTONDOWN:
        {
            vector<cv::Point2f> control_points = g_roi.getControlPoints();
            g_selected_point = findClickedPoint(g_mouse_pos, control_points);
            g_dragging = (g_selected_point != NONE);
            break;
        }
        
        case cv::EVENT_MOUSEMOVE:
        {
            if (g_dragging && g_selected_point != NONE) {
                switch (g_selected_point) {
                    case TL:  // Move entire ROI
                    {
                        int dx = g_mouse_pos.x - g_roi.tl.x;
                        int dy = g_mouse_pos.y - g_roi.tl.y;
                        g_roi.br.x += dx;
                        g_roi.br.y += dy;
                        g_roi.tl = g_mouse_pos;
                        break;
                    }
                    
                    case TR:  // Rotate around center
                    {
                        cv::Point2f center = g_roi.center();
                        cv::Point2f old_tr = cv::Point2f(g_roi.br.x, g_roi.tl.y);
                        
                        // Calculate angle between old and new top-right point relative to center
                        double old_angle = atan2(old_tr.y - center.y, old_tr.x - center.x) * 180.0 / M_PI;
                        double new_angle = atan2(g_mouse_pos.y - center.y, g_mouse_pos.x - center.x) * 180.0 / M_PI;
                        
                        g_roi.angle = new_angle - old_angle;
                        break;
                    }
                    
                    case L:  // Move left edge
                    {
                        g_roi.tl.x = x;
                        // Ensure ROI has positive width
                        if (g_roi.tl.x >= g_roi.br.x) {
                            g_roi.tl.x = g_roi.br.x - 1;
                        }
                        break;
                    }
                    
                    case R:  // Move right edge
                    {
                        g_roi.br.x = x;
                        // Ensure ROI has positive width
                        if (g_roi.br.x <= g_roi.tl.x) {
                            g_roi.br.x = g_roi.tl.x + 1;
                        }
                        break;
                    }
                    
                    case T:  // Move top edge
                    {
                        g_roi.tl.y = y;
                        // Ensure ROI has positive height
                        if (g_roi.tl.y >= g_roi.br.y) {
                            g_roi.tl.y = g_roi.br.y - 1;
                        }
                        break;
                    }
                    
                    case B:  // Move bottom edge
                    {
                        g_roi.br.y = y;
                        // Ensure ROI has positive height
                        if (g_roi.br.y <= g_roi.tl.y) {
                            g_roi.br.y = g_roi.tl.y + 1;
                        }
                        break;
                    }
                    
                    default:
                        break;
                }
                
                // Redraw
                g_img_copy = g_img.clone();
                drawROI(g_img_copy, g_roi);
                cv::imshow("Template Creator", g_img_copy);
            }
            break;
        }
        
        case cv::EVENT_LBUTTONUP:
        {
            g_dragging = false;
            break;
        }
        
        default:
            break;
    }
}

int main(int argc, char** argv) {
    string img_path;
    bool auto_save = false;
    
    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--auto-save") {
            auto_save = true;
        } else {
            img_path = arg;
        }
    }
    
    // If no image path provided, ask for it
    if (img_path.empty()) {
        cout << "Enter image path: ";
        cin >> img_path;
    }
    
    g_img = cv::imread(img_path);
    if (g_img.empty()) {
        cerr << "Failed to open image!" << endl;
        return -1;
    }
    
    g_img_copy = g_img.clone();
    
    // Initialize ROI
    int init_width = 200;
    int init_height = 200;
    g_roi = ROI(cv::Point(g_img.cols / 2 - init_width / 2, g_img.rows / 2 - init_height / 2),
                cv::Point(g_img.cols / 2 + init_width / 2, g_img.rows / 2 + init_height / 2));
    
    // Create window
    cv::namedWindow("Template Creator", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Template Creator", mouseCallback, NULL);
    
    // Draw initial ROI
    drawROI(g_img_copy, g_roi);
    cv::imshow("Template Creator", g_img_copy);
    
    cout << "Instructions:" << endl;
    cout << "1. Click and drag top-left corner (red) to move entire ROI" << endl;
    cout << "2. Click and drag top-right corner (blue) to rotate ROI" << endl;
    cout << "3. Click and drag edge centers (green) to resize ROI" << endl;
    cout << "4. Press 's' to save ROI as template" << endl;
    cout << "5. Press 'q' to quit" << endl;
    
    // If auto-save option is enabled, save and exit
    if (auto_save) {
        saveROI(g_roi, "template_roi.txt");
    } else {
        while (true) {
            int key = cv::waitKey(10);
            
            if (key == 'q' || key == 27) {  // 'q' or ESC to quit
                break;
            } else if (key == 's') {  // 's' to save ROI
                saveROI(g_roi, "template_roi.txt");
            }
        }
    }
    
    cv::destroyAllWindows();
    return 0;
}
