import logging

from app.db.database_manager import Activity


class ActivityDataHandler:

    def __init__(self, user_id):
        self.user_id = user_id

    def fetch_recovery_data(self):
        activities = Activity.query.filter_by(
            user_id=self.user_id,
            activity_type="recovery",
        ).all()

        if not activities:
            logging.warning("No recovery activities found for the user.")
            return {}

        activity_names = ", ".join(
            set(
                activity.activity_name
                for activity in activities
                if activity.activity_name
            )
        )
        frequencies = ", ".join(
            set(
                activity.frequency
                for activity in activities
                if activity.frequency
            )
        )
        fun_levels = ", ".join(
            set(
                str(activity.fun_level)
                for activity in activities
                if activity.fun_level is not None
            )
        )

        return {
            "activity_name": activity_names,
            "frequency": frequencies,
            "fun_level": fun_levels,
        }

    def fetch_social_data(self):
        activities = Activity.query.filter_by(
            user_id=self.user_id,
            activity_type="social",
        ).all()

        if not activities:
            logging.warning("No social activities found for the user.")
            return {}

        done_with_values = "|".join(
            set(
                value.strip()
                for activity in activities
                if activity.done_with
                for value in activity.done_with.split("|")
            )
        )

        activity_names = ", ".join(
            set(
                activity.activity_name
                for activity in activities
                if activity.activity_name
            )
        )

        frequencies = ", ".join(
            set(
                activity.frequency
                for activity in activities
                if activity.frequency
            )
        )

        return {
            "activity_name": activity_names,
            "frequency": frequencies,
            "done_with": done_with_values,
        }

    def fetch_important_data(self):
        activities = Activity.query.filter(
            Activity.user_id == self.user_id,
            Activity.importance >= 4,
        ).all()

        if not activities:
            logging.warning("No important activities found for the user.")
            return {}

        activity_names = ", ".join(
            set(
                activity.activity_name
                for activity in activities
                if activity.activity_name
            )
        )

        frequencies = ", ".join(
            set(
                activity.frequency
                for activity in activities
                if activity.frequency
            )
        )

        importance_values = ", ".join(
            str(activity.importance)
            for activity in activities
            if activity.importance is not None
        )

        return {
            "activity_name": activity_names,
            "importance": importance_values,
            "frequency": frequencies,
        }

    def fetch_fun_data(self):
        query = Activity.query.filter_by(user_id=self.user_id)

        top = query.filter(Activity.fun_level == 5).all()

        if not top:
            top = query.filter(Activity.fun_level >= 4).all()

        if not top:
            top = query.filter(Activity.fun_level.isnot(None)).all()

        if not top:
            logging.warning("No fun-like activities found for the user.")
            return {}

        activity_names = ", ".join(
            {
                activity.activity_name
                for activity in top
                if activity.activity_name
            }
        )

        frequencies = ", ".join(
            {
                activity.frequency
                for activity in top
                if activity.frequency
            }
        )

        fun_levels = ", ".join(
            {
                str(activity.fun_level)
                for activity in top
                if activity.fun_level is not None
            }
        )

        return {
            "activity_name": activity_names,
            "fun_level": fun_levels or "N/A",
            "frequency": frequencies,
        }

    def fetch_motivation_data(self):
        logging.info(
            "Fetching motivated activities for user_id: %s",
            self.user_id,
        )

        activities = Activity.query.filter(
            Activity.user_id == self.user_id,
            Activity.motivation.isnot(None),
            Activity.motivation != "",
        ).all()

        if not activities:
            logging.warning("No motivated activities found for the user.")
            return {}

        motivations = "|".join(
            set(
                motivation.strip()
                for activity in activities
                if activity.motivation
                for motivation in activity.motivation.split("|")
            )
        )

        frequencies = ", ".join(
            set(
                activity.frequency
                for activity in activities
                if activity.frequency
            )
        )

        return {
            "motivation": motivations,
            "frequency": frequencies,
        }

    def fetch_priority_mix(self, top_n=2):
        # Select top candidates by ratings only.
        top_activities = Activity.query.filter(
            Activity.user_id == self.user_id,
            Activity.importance.isnot(None),
            Activity.fun_level.isnot(None),
        ).order_by(
            Activity.importance.desc(),
            Activity.fun_level.desc(),
        ).limit(top_n).all()

        lower_priority_activities = Activity.query.filter(
            Activity.user_id == self.user_id,
        ).filter(
            (Activity.importance <= 2)
            | (Activity.fun_level <= 2)
        ).limit(1).all()

        selected_activities = (
            (top_activities or [])
            + (lower_priority_activities or [])
        )

        if not selected_activities:
            return {}

        # Keep activity attributes paired for use in prompts.
        rows = []

        for activity in selected_activities:
            rows.append(
                {
                    "activity_name": activity.activity_name or "",
                    "importance": (
                        activity.importance
                        if activity.importance is not None
                        else ""
                    ),
                    "fun_level": (
                        activity.fun_level
                        if activity.fun_level is not None
                        else ""
                    ),
                    "frequency": activity.frequency or "",
                }
            )

        # Flat strings retained for compatibility with coach_agent.
        activity_names = ", ".join(
            {
                row["activity_name"]
                for row in rows
                if row["activity_name"]
            }
        )

        importance_values = ", ".join(
            str(row["importance"])
            for row in rows
            if row["importance"] != ""
        )

        fun_levels = ", ".join(
            str(row["fun_level"])
            for row in rows
            if row["fun_level"] != ""
        )

        frequencies = ", ".join(
            {
                row["frequency"]
                for row in rows
                if row["frequency"]
            }
        )

        return {
            "activity_name": activity_names,
            "importance": importance_values,
            "fun_level": fun_levels,
            "frequency": frequencies,
            "rows": rows,
        }