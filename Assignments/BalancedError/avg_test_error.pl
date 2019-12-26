#!/usr/bin/perl

$mean = 0;
$data = shift;
#$dir=$data;
$dir=$data;
for(my $i=1; $i<2; $i++){
  system("python3 least_squares_adaptive_eta.py $dir/$data.data $dir/$data.trainlabels.$i > nm_out.$data");
  $err[$i] = `perl error.pl $dir/$data.labels nm_out.$data`;
  chomp $err[$i];
  print "$err[$i]\n";
  $mean += $err[$i];
  print "$mean\n";
}
$mean *= 100;
$sd = 0;
for(my $i=1; $i<2; $i++){
  $sd += ($err[$i]-$mean)**2;
}
$sd /= 1;
$sd = sqrt($sd);
print "adaptive eta Least sqaure error = $mean% ($sd)\n";

$mean = 0;
for(my $i=1; $i<2; $i++){
  system("python3 hinge_adaptive_eta.py $dir/$data.data $dir/$data.trainlabels.$i > nm_out.$data");
  $err[$i] = `perl error.pl $dir/$data.labels nm_out.$data`;
  chomp $err[$i];
  print "$err[$i]\n";
  $mean += $err[$i];
  #print "$mean\n";
}
$mean *= 100;
$sd = 0;
for(my $i=1; $i<2; $i++){
  $sd += ($err[$i]-$mean)**2;
}
$sd /= 1;
$sd = sqrt($sd);
print "adaptive eta Hinge error = $mean% ($sd)\n";
