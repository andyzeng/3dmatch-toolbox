# Make html file with class documentation
# NOTE: Does not insert inherited members

$0~"class " && $0~"{" {
  # Parse package name and class name
  pkg_name = "RNFoobar";
  class_name = $2;

  # Print class name
  printf "<HR> <H2> <A NAME=\"%s\" HREF=\"../pkgs/%s/%s.h\">%s</A> </H2> \n\n",
    class_name, pkg_name, class_name, class_name;

  # Print base classes
  printf "<DL> <DL>\n";
  printf "<DT> <H3>Base Classes:</H3> \n";
  printf "<DL>\n";
  printf "<DT> <H4>Public Base Classes:</H4> \n";
  printf "<DL>\n";
  found = 0;
  split($0, s0);
  for (i = 3; i <= NF; i++) {
    if (s0[i] == "public") {
      printf "<DT>    <A HREF=\"#%s\">%s</A>\n", $(i+1), $(i+1)
      found = 1;
    }
  }
  if (!found) printf "<DT>\tNone\n";
  printf "</DL> </DL>\n";
  printf "<P>\n";

  # Print member functions
  printf "<DT> <H3>Member Functions:</H3>\n";
  printf "<DL>\n";
  active = 0;
  first = 1;
}

$0~"};" {
  if (!first) printf "</DL>\n";
  printf "</DL> </DL>\n";
  printf "</DL>\n\n\n\n";
  printf "<P>\n";
  active = 0;
}

{ 
  sub("\r$", ""); 
  if ($1 == "public:") {
    active = 1;
  }
  else if (($1 == "protected:") || ($1 == "private:")) {
    active = 0;
  }
  else if (active) {
    if ($1 == "//") {
      split($0, s0)
      if (!first) printf "</DL>\n";
      printf "<DT> <H4>";
      for (i = 2; i <= NF; i++) printf " %s", s0[i];
      printf ": </H4>\n";
      printf "<DL>\n";
      first = 0;
    }
    else {
	print "<DT>" $0;
    }
  }  
}

