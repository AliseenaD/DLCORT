# Function that checks to see if nose is within radius and pointing at the center of an object
    def check_interaction(self, nose_cords, left_cords, right_cords, object_regions):
        nx, ny = nose_cords
        lx, ly = left_cords
        rx, ry = right_cords
        # Calculate midpoint between ears
        mid_x = (lx + rx) / 2
        mid_y = (ly + ry) / 2
        # Calculate nose direction vector (from midpoint to nose)
        nose_vector = np.array([nx - mid_x, ny - mid_y])
        # Calculate the angle change per pixel so that angle of nose can adjust based on distance
        angle_diff = 90 - float(self.angle.get())
        angle_per_pixel = angle_diff / float(self.radius.get())

        # Iterate through each object and check to see if nose is within radius passed in
        for object in object_regions:
            ox, oy = object["center"]
            distance = math.sqrt((ox - nx)**2 + (oy - ny)**2)
            # If within radius then check angle of nose as well
            if distance < object['radius']:
                # Create vector from nose to center of object
                object_vector = np.array([ox - nx, oy - ny])
                # Find difference in distance between radius and actual distance of nose
                distance_difference = float(self.radius.get()) - distance
                # Check if angle of head is within angle of threshold given the distance to object
                if self.calculate_angle_between_vectors(nose_vector, object_vector) <= distance_difference * angle_per_pixel + float(self.angle.get()):
                    # Increment the interaction count of the object
                    object["interaction_count"] += 1
                    return object["title"]
        return None