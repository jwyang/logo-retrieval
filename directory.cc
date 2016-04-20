#include <boost/filesystem.hpp>
#include <boost/filesystem/v3/operations.hpp>
#include <boost/filesystem/v3/path.hpp>
#include <dirent.h>
#include <stddef.h>
#include <string>
#include <vector>

using namespace boost::filesystem;
using std::string;
using std::vector;

namespace fileutil {

bool GetDir (string dir, vector<string> *files) {
  DIR *dp;
  struct dirent *dirp;
  if((dp  = opendir(dir.c_str())) == NULL) {
    return false;
  }
  while ((dirp = readdir(dp)) != NULL) {
    string name = string(dirp->d_name);
    if (name.length() > 2)
      files->push_back(name);
  }
  closedir(dp);
  return true;
}

bool GetDirRecursive (string dir, vector<string> *files) {
  DIR *dp;
  struct dirent *dirp;
  if((dp  = opendir(dir.c_str())) == NULL) {
    return false;
  }
  while ((dirp = readdir(dp)) != NULL) {
    string name = string(dirp->d_name);
    if (name != "." && name != "..") {
      path p(dir + "/" + name);
      if (is_directory(p)) {
	GetDirRecursive(dir + "/" + name, files);
      } else {
	files->push_back(dir + "/" + name);
      }
    }
  }

  closedir(dp);
  return true;
}

bool GetDirRecursive_Spec (string dir, string ftype, vector<string> *files) {
  DIR *dp;
  struct dirent *dirp;
  if((dp  = opendir(dir.c_str())) == NULL) {
    return false;
  }
  while ((dirp = readdir(dp)) != NULL) {
    string name = string(dirp->d_name);
    if (name != "." && name != "..") {
      path p(dir + "/" + name);
      if (is_directory(p)) {
	GetDirRecursive_Spec(dir + "/" + name, ftype, files);
      } else {
        if (name.find(ftype) > 0 && name.find(ftype) < name.length())
          files->push_back(dir + "/" + name);
      }
    }
  }

  closedir(dp);
  return true;
}

}
