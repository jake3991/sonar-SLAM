sim_scene="waterfront" #plane, penns_landing, suny or rfal_land
echo "running ""$sim_scene"" simulation grid search"
for kf_translation in 1 2 3 4 5
do
    for kf_rotation in 30 60 90
    do 
        echo $kf_translation" "$kf_rotation

        if [ "$sim_scene" == "suny_pier" ]
        then
            roslaunch bruce_slam slam.launch file:="/home/jake/Desktop/real_bags/deep_2020-08-19-13-51-45.bag" duration:=385 rotation:=$kf_rotation translation:=$kf_translation scene:=$sim_scene &
            sleep 700
        fi

        if [ "$sim_scene" == "waterfront" ]
        then
            roslaunch bruce_slam slam.launch file:="/home/jake/Desktop/real_bags/wtr4_2020-08-19-13-28-21.bag" start:=260 end:=520 rotation:=$kf_rotation translation:=$kf_translation scene:=$sim_scene &
            sleep 630
        fi
        
        rosnode kill -a
        killall -9 rosmaster
        killall -9 roscore
        sleep 10
    done
done


