#ifndef UTILS_FILEUTIL_DIRECTORY_H__
#define UTILS_FILEUTIL_DIRECTORY_H__

using std::string;
using std::vector;

namespace fileutil {

  // Get the files under the directory
  bool GetDir (string dir, vector<string> *files);

  // Get the files recursively under the directory
  bool GetDirRecursive (string dir, vector<string> *files);

  // Get the files recursively under the directory with some specific file type
  bool GetDirRecursive_Spec (string dir, string ftype, vector<string> *files);
}


#endif // UTILS_FILEUTIL_DIRECTORY_H__
