sim_scene="plane" #plane, penns_landing or rfal_land
echo "running ""$sim_scene"" simulation grid search"
for kf_translation in 1 2 3 4 5
do
    for kf_rotation in 30 60 90
    do 
        echo $kf_translation" "$kf_rotation
        roslaunch bruce_slam slam.launch file:="/home/jake/Desktop/sim_bags/"$sim_scene".bag" rotation:=$kf_rotation translation:=$kf_translation scene:=$sim_scene &

        if [ "$sim_scene" == "plane" ]
        then
            sleep 360
        fi

        if [ "$sim_scene" == "rfal_land" ]
        then
            sleep 320
        fi

        if [ "$sim_scene" == "penns_landing" ]
        then
            sleep 1140
        fi
        
        rosnode kill -a
        killall -9 rosmaster
        killall -9 roscore
        sleep 10
    done
done


