import json
peoples=[]
with open('C:/Users/123/Desktop/PoseKeypoint_00001.json') as fr:
    peoples=json.load(fr)

trajs=[]

frame_length=100
for ipose in range(int(len(peoples[0]["people"][0]["pose_keypoints_2d"])/3)):
    traj=[]
    for peoplej in peoples:
        if len(traj)<frame_length-1:
            people=peoplej["people"]
            if people[0]["pose_keypoints_2d"][ipose*3+2]==1.0:
                x=people[0]["pose_keypoints_2d"][ipose*3]
                y=people[0]["pose_keypoints_2d"][ipose*3+1]

                if x<=576 and y<=320:
                    traj.append([x,y])
                else:
                    break
            else:
                if len(traj)>0:
                    traj.append(traj[len(traj)-1])
                else:
                    break
    if len(traj)>0:
        trajs.append(traj)
    
with open("C:/Users/123/Desktop/traj_00001.json", "w") as fw:
    json.dump(trajs,fw)