import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors={}
        self.player_team_dict={}
        self.kmeans = None

    def _crop_player_image(self, frame, bbox):
        frame_height, frame_width = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(frame_width, int(bbox[2]))
        y2 = min(frame_height, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            return None

        image = frame[y1:y2, x1:x2]
        if image.size == 0:
            return None

        top_half_height = max(1, image.shape[0] // 2)
        return image[:top_half_height, :]


    def get_clustering_model(self,image):
        # Reshape the image to 2D array
        image_2d = image.reshape(-1,3)

        #Performs K-means with 2 clusters
        kmeans= KMeans(n_clusters=2, init="k-means++",n_init=3, random_state=0)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self,frame,bbox):
        top_half_image = self._crop_player_image(frame, bbox)
        if top_half_image is None or top_half_image.shape[0] < 2 or top_half_image.shape[1] < 2:
            return None

        # Get Clustering model
        kmeans =self.get_clustering_model(top_half_image)

        # Get cluster labels for each pixel
        labels = kmeans.labels_

        #Reshape the labels to image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        #Get the player cluster 
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1- non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self,frames,player_tracks,sample_frames=10):
        player_colors=[]
        frames_to_sample = min(sample_frames, len(frames), len(player_tracks))

        for frame_num in range(frames_to_sample):
            frame = frames[frame_num]
            for _, player_detection in player_tracks[frame_num].items():
                bbox=player_detection["bbox"]
                player_color= self.get_player_color(frame,bbox)
                if player_color is not None:
                    player_colors.append(player_color)

        if len(player_colors) < 2:
            self.team_colors[1] = np.array([255, 0, 0])
            self.team_colors[2] = np.array([0, 0, 255])
            self.kmeans = None
            return

        kmeans = KMeans(n_clusters=2, init="k-means++",n_init=10, random_state=0)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1]=kmeans.cluster_centers_[0]
        self.team_colors[2]=kmeans.cluster_centers_[1]

    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            team_id = 1 if len(self.player_team_dict) % 2 == 0 else 2
            self.player_team_dict[player_id]=team_id
            return team_id

        player_color = self.get_player_color(frame,player_bbox)
        if player_color is None:
            team_id = 1 if len(self.player_team_dict) % 2 == 0 else 2
            self.player_team_dict[player_id]=team_id
            return team_id

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        self.player_team_dict[player_id]=team_id

        return team_id




